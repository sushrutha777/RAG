"""LangGraph nodes for RAG workflow with custom ReAct agent."""
import json
import re
import time

from typing import List
from src.state.rag_state import RAGState
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage


class RAGNodes:
    """Contains node functions for RAG workflow."""

    def __init__(self, retriever, llm):
        self.retriever = retriever      # VectorStoreRetriever
        self.llm = llm                  # Chat model
        self.tools = {}

    # ── Retry wrapper for rate-limit resilience ───────────────
    def _invoke_with_retry(self, prompt, max_retries=3):
        """Invoke LLM with exponential backoff on 429 / rate-limit errors."""
        for attempt in range(max_retries):
            try:
                return self.llm.invoke(prompt)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = (
                    "429" in err
                    or "rate" in err
                    or "resource_exhausted" in err
                    or "resource exhausted" in err
                    or "too many requests" in err
                )
                if is_rate_limit and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)        # 2s, 4s, 8s
                    print(f"Rate limited — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                raise                                # non-rate-limit → bubble up
        raise Exception("Max retries exceeded due to rate limiting")

    # helpers
    def _format_history(self, state: RAGState) -> str:
        """Format prior messages into a readable conversation history string.
        
        Excludes the current (last) HumanMessage so the caller can
        place it separately in the prompt.
        """
        if not state.messages or len(state.messages) <= 1:
            return "No prior conversation."
        
        lines = []
        for msg in state.messages[:-1]:          # skip current question
            role = "User" if msg.type == "human" else "Assistant"
            # Truncate very long messages to keep prompts manageable
            content = msg.content[:800] if isinstance(msg.content, str) else str(msg.content)[:800]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # 0. REWRITE QUERY NODE — resolve pronouns / follow-ups
    def rewrite_query(self, state: RAGState) -> RAGState:
        """Rewrite the user question into a standalone query using conversation history.

        If there is no prior conversation, the question is returned as-is.
        This ensures pronouns like 'he', 'she', 'it', 'they' are resolved
        before the retriever or web search sees the query.
        """
        history = self._format_history(state)

        # Nothing to rewrite when there's no prior conversation
        if history == "No prior conversation.":
            return state

        rewrite_prompt = (
            "You are a query rewriter. Your ONLY job is to rewrite the user's "
            "follow-up question into a fully standalone question that does not "
            "rely on any conversation context.\n\n"
            "Rules:\n"
            "1. Replace ALL pronouns (he, she, it, they, him, her, etc.) with "
            "the actual entity they refer to from the conversation history.\n"
            "2. Keep the rewritten question concise and natural.\n"
            "3. Output ONLY the rewritten question — no explanation, no quotes, "
            "no extra text.\n"
            "4. If the question is already standalone, return it unchanged.\n\n"
            f"Conversation history:\n{history}\n\n"
            f"Follow-up question: {state.question}\n\n"
            "Rewritten standalone question:"
        )

        try:
            msg = self._invoke_with_retry(rewrite_prompt)
            rewritten = getattr(msg, "content", str(msg)).strip()
            # Only accept if the model actually returned something useful
            if rewritten and len(rewritten) > 2:
                state.question = rewritten
        except Exception:
            pass  # keep original question on failure

        return state

    # 1. RETRIEVE DOCUMENTS NODE
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve relevant docs for the given question."""
        docs = self.retriever.invoke(state.question)  
        state.retrieved_docs = docs
        return state

    # 1.5 DECISION ENGINE NODE
    def decision_engine(self, state: RAGState) -> RAGState:
        """
        Specialized decision node to evaluate if retrieved docs are sufficient,
        if more context (web) is needed, or if the query is out of scope.
        """
        # Prepare summaries (top-k, short)
        doc_summaries = []
        for i, d in enumerate(state.retrieved_docs[:5], start=1):
            content = d.page_content[:200].replace("\n", " ").strip()
            doc_summaries.append(f"Doc {i}: {content}...")
        
        doc_summaries_str = "\n".join(doc_summaries) if doc_summaries else "No documents retrieved."

        prompt = (
            "You are a strict decision engine.\n\n"
            "Context:\n"
            "- The documents are already chunked and indexed.\n"
            "- You will NOT ask follow-up questions.\n"
            "- You will NOT request more context.\n"
            "- You will make ONE decision only.\n\n"
            "Input:\n"
            f"User query:\n{state.question}\n\n"
            "Retrieved document summaries (top-k, short):\n"
            f"{doc_summaries_str}\n\n"
            "Task:\n"
            "Decide the best action to take.\n\n"
            "Rules:\n"
            "- Choose exactly ONE option from the list below.\n"
            "- Do NOT explain your reasoning.\n"
            "- Do NOT add extra text.\n"
            "- Do NOT format as markdown.\n"
            "- Output must be valid JSON only.\n\n"
            "Options:\n"
            "1. \"answer_from_documents\" – if the documents clearly contain the answer\n"
            "2. \"need_more_context\" – if the documents are insufficient\n"
            "3. \"out_of_scope\" – if the query is unrelated to the documents\n\n"
            "Output format:\n"
            "{\n"
            "  \"decision\": \"<one_option>\",\n"
            "  \"confidence\": <number between 0 and 1>\n"
            "}\n"
        )

        try:
            msg = self._invoke_with_retry(prompt)
            content = getattr(msg, "content", str(msg)).strip()
            # Clean possible markdown formatting if the model ignored instructions
            if content.startswith("```"):
                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```\s*", "", content)
            
            data = json.loads(content)
            state.decision = data.get("decision", "need_more_context")
            state.confidence = data.get("confidence", 0.0)
        except Exception as e:
            # Fallback
            state.decision = "need_more_context"
            state.confidence = 0.0
        
        return state

    # 2. BUILD TOOLSET (retriever + wikipedia + websearch)
    def _build_tools(self):
        from langchain_community.utilities import WikipediaAPIWrapper
        from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
        from langchain_core.documents import Document

        # RETRIEVER TOOL
        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
    
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = getattr(d, "metadata", {})
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
    
            return "\n\n".join(merged)
    
        retriever_tool = Tool(
            name="retriever",
            description="Search indexed corpus for relevant text.",
            func=retriever_tool_fn,
        )
    
        # WIKIPEDIA TOOL
        wiki_api = WikipediaAPIWrapper(top_k_results=3, lang="en")
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge.",
            func=wiki_api.run,
        )

        # DUCKDUCKGO WEB SEARCH TOOL
        ddg = DuckDuckGoSearchRun()
    
        websearch_tool = Tool(
            name="web_search",
            description="Unlimited web search using DuckDuckGo.",
            func=ddg.run,
        )
        # REGISTER ALL TOOLS
        self.tools = {
            "retriever": retriever_tool,
            "wikipedia": wikipedia_tool,
            "web_search": websearch_tool,
        }
        
    # 3. EXECUTE A TOOL BY NAME
    def _run_tool(self, name: str, input: str) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"Tool '{name}' does not exist."
        try:
            result = tool.func(input)
            # Guard against empty/None results from tools
            if not result or (isinstance(result, str) and not result.strip()):
                return f"Tool '{name}' returned no results for: {input}"
            return result
        except Exception as e:
            return f"Tool '{name}' failed: {e}"

    # 4. GENERATE ANSWER WITH CUSTOM REACT LOOP
    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using a manual ReAct loop (with web_search for latest info)."""
        question = state.question

        # Build conversation history for context continuity
        history_context = self._format_history(state)

        if not self.tools:
            self._build_tools()

        # detect time-sensitive / "latest" style questions 
        q_lower = question.lower()
        time_sensitive_keywords = [
            "latest", "today", "yesterday", "current", "now", "recent", "this year",
            "last week", "last month", "this month", "this week",
            "winner", "champion", "election", "price", "score", "result",
            "who is", "who won", "what happened", "who was",
            "breaking", "news", "update", "announced", "launched",
            "attacked", "war", "2023", "2024", "2025", "2026", "2027"
        ]
        is_time_sensitive = any(k in q_lower for k in time_sensitive_keywords)

        # General knowledge questions also benefit from web_search (more reliable than wikipedia API)
        general_knowledge_keywords = [
            "where was", "where did", "where is", "where does",
            "when was", "when did", "when is",
            "how old", "how tall", "how much", "how many",
            "born", "died", "age", "height", "weight",
            "capital of", "population of", "founder of", "ceo of",
            "president of", "prime minister",
        ]
        is_general_knowledge = any(k in q_lower for k in general_knowledge_keywords)

        # Prefer web_search for time-sensitive OR general knowledge queries
        forced_tool = None
        if "web_search" in self.tools and (is_time_sensitive or is_general_knowledge):
            forced_tool = "web_search"

        # Step 1: Ask LLM what to do (ReAct decision) 
        think_prompt = (
            "You are a ReAct-style agent with access to tools.\n"
            "Available tools:\n"
            "  - retriever: search the internal indexed corpus.\n"
            "  - wikipedia: general encyclopedic knowledge (may be slightly outdated).\n"
            "  - web_search: real-time web search, always up-to-date.\n\n"
            "CRITICAL RULES:\n"
            "1. If the question involves recent events, sports results, elections, champions, "
            "   financial markets, 'latest', 'current', 'today','yesterday' or specific years like 2023/2024/2025/2026/2027,\n"
            "   you MUST use the 'web_search' tool.\n"
            "2. Only answer directly with no tool if you are VERY sure the answer is timeless.\n"
            "3. Respond EXACTLY in this JSON format (no extra text):\n\n"
            "{\n"
            '  "tool": "<retriever | wikipedia | web_search | none>",\n'
            '  "input": "<STANDALONE search query. You MUST resolve all pronouns using the Conversation history>"\n'
            "}\n\n"
            f"Conversation history:\n{history_context}\n\n"
            f"User question: {question}"
        )

        tool = None
        tool_input = question

        # If we already decided it's time-sensitive, we can skip LLM decision if you want.
        # But better: still let LLM choose input phrasing, while forcing the tool.
        if forced_tool:
            try:
                decision_msg = self._invoke_with_retry(think_prompt)
                decision_text = getattr(decision_msg, "content", str(decision_msg)).strip()
                decision_text = re.sub(r"^```(?:json)?\s*", "", decision_text, flags=re.IGNORECASE)
                decision_text = re.sub(r"\s*```$", "", decision_text).strip()
                parsed = json.loads(decision_text)
                tool_input = parsed.get("input") or question
            except Exception:
                m_in = re.search(r'"input"\s*:\s*"([^"]+)"', decision_text) if 'decision_text' in locals() else None
                tool_input = m_in.group(1) if m_in else question
            tool = forced_tool  # override whatever the model chose
        else:
            # Normal path: let LLM decide tool & input
            try:
                decision_msg = self.llm.invoke(think_prompt)
                decision_text = getattr(decision_msg, "content", str(decision_msg)).strip()
            except Exception as e:
                state.answer = f"LLM error during decision step: {e}"
                return state

            decision_text = re.sub(r"^```(?:json)?\s*", "", decision_text, flags=re.IGNORECASE)
            decision_text = re.sub(r"\s*```$", "", decision_text).strip()

            # Step 1b: Robust JSON parsing fallback 
            try:
                parsed = json.loads(decision_text)
                tool = (parsed.get("tool") or "none").lower()
                if parsed.get("input"):
                    tool_input = parsed["input"]
            except Exception:
                # Try to extract a tool name with regex as a fallback
                m_tool = re.search(r'"tool"\s*:\s*"([^"]+)"', decision_text)
                tool = m_tool.group(1).lower() if m_tool else "none"
                m_in = re.search(r'"input"\s*:\s*"([^"]+)"', decision_text)
                tool_input = m_in.group(1) if m_in else question

        # Step 2: Run selected tool (if any)
        tool_result = ""
        used_tool_name = "none"

        if tool and tool != "none" and tool.lower() in self.tools:
            used_tool_name = tool.lower()
            tool_result = self._run_tool(used_tool_name, tool_input)

            # Fallback: if result is empty/unhelpful and we didn't already use web_search, retry
            no_result_indicators = [
                "no results", "no documents found", "returned no results",
                "Tool '", "could not find", "no information",
            ]
            if (
                used_tool_name != "web_search"
                and "web_search" in self.tools
                and any(ind.lower() in tool_result.lower() for ind in no_result_indicators)
            ):
                fallback_result = self._run_tool("web_search", tool_input)
                if fallback_result and not any(
                    ind.lower() in fallback_result.lower() for ind in no_result_indicators
                ):
                    tool_result = fallback_result
                    used_tool_name = "web_search (fallback)"
        else:
            # No tool: we'll answer from prior knowledge only
            tool_result = "No external tool was used. Answer from your own knowledge."

        # Step 3: Final Answer Synthesis — with conversation history
        final_prompt = (
            "You are a helpful, grounded question-answering assistant.\n\n"
            "Your task is to answer the user's question using the provided context.\n\n"
            "Context:\n"
            f"{tool_result}\n\n"
            "Conversation History:\n"
            f"{history_context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "RULES:\n\n"
            "1. Use information from the provided context to answer the question.\n"
            "2. If the context contains relevant information, extract and present it clearly and concisely.\n"
            "3. DO NOT invent specific facts, statistics, dates, or quotes that are not in the context.\n"
            "4. If the context partially answers the question, provide what you can from the context.\n"
            "5. Only if the context contains absolutely NO relevant information, say:\n"
            "   \"I could not find reliable information about that. Try rephrasing your question.\"\n"
            "6. Do NOT mention that you are an AI model.\n"
            "7. Keep the answer concise and factual.\n"
            "8. If the context has conflicting information from different sources, mention the conflict rather than picking one.\n\n"
            "Answer:\n"
        )

        try:
            final_msg = self._invoke_with_retry(final_prompt)
            answer = getattr(final_msg, "content", str(final_msg)).strip()
            
            if not answer:
                answer = ("I'm sorry, I don't have specific information about that right now. "
                          "Please try rephrasing your question or ask something else.")

            # Append tool usage info
            if used_tool_name and used_tool_name != "none":
                answer += f"\n\n(Tool Used: {used_tool_name})"
        except Exception as e:
            answer = f"LLM error during final answer: {e}"

        state.answer = answer

        # Persist the assistant reply in messages for memory continuity
        state.messages.append(AIMessage(content=answer))

        return state
