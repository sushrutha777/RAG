"""LangGraph nodes for RAG workflow with custom ReAct agent."""
import json
import re

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
            msg = self.llm.invoke(prompt)
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
            "latest", "today", "yesterday", "current", "now", "recent", "this year","last week", "last month", "this month", "this week",
            "winner", "champion", "election", "price", "score", "result",
            "who is", "who won", "what happened", "who was",
            "breaking", "news", "update", "announced", "launched",
            "attacked", "war", "2023", "2024", "2025", "2026", "2027"
        ]
        is_time_sensitive = any(k in q_lower for k in time_sensitive_keywords)

        # If clearly time-sensitive, strongly bias towards web_search
        forced_tool = None
        if "web_search" in self.tools and is_time_sensitive:
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
            '  "input": "<tool input or empty if none>"\n'
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
                decision_msg = self.llm.invoke(think_prompt)
                decision_text = getattr(decision_msg, "content", str(decision_msg)).strip()
                parsed = json.loads(decision_text)
                tool_input = parsed.get("input") or question
            except Exception:
                tool_input = question
            tool = forced_tool  # override whatever the model chose
        else:
            # Normal path: let LLM decide tool & input
            try:
                decision_msg = self.llm.invoke(think_prompt)
                decision_text = getattr(decision_msg, "content", str(decision_msg)).strip()
            except Exception as e:
                state.answer = f"LLM error during decision step: {e}"
                return state

            # Step 1b: Robust JSON parsing fallback 
            try:
                parsed = json.loads(decision_text)
                tool = (parsed.get("tool") or "none").lower()
                if parsed.get("input"):
                    tool_input = parsed["input"]
            except Exception:
                # Try to extract a tool name with regex as a fallback
                m = re.search(r'"tool"\s*:\s*"([^"]+)"', decision_text)
                tool = m.group(1).lower() if m else "none"
                tool_input = question

        # Step 2: Run selected tool (if any)
        tool_result = ""
        used_tool_name = "none"

        if tool and tool != "none" and tool.lower() in self.tools:
            used_tool_name = tool.lower()
            tool_result = self._run_tool(used_tool_name, tool_input)
        else:
            # No tool: we'll answer from prior knowledge only
            tool_result = "No external tool was used. Answer from your own knowledge."

        # Step 3: Final Answer Synthesis — with conversation history
        final_prompt = (
            "You are an expert assistant.\n\n"
            f"Conversation history:\n{history_context}\n\n"
            f"User question:\n{question}\n\n"
            f"Tool chosen: {used_tool_name}\n"
            f"Tool input: {tool_input}\n\n"
            "Tool result (may be empty or noisy, but do your best to extract the answer):\n"
            f"{tool_result}\n\n"
            "Instructions for your final answer:\n"
            "1. If the tool_result clearly contains the answer, use it and explain concisely.\n"
            "2. If the tool_result is empty or unhelpful, you may answer from your own knowledge, "
            "   but make it clear it's based on general knowledge and may be outdated.\n"
            "3. Do NOT mention that you are using tools, ReAct, or internal reasoning.\n"
            "4. Just give a natural, direct answer for the user.\n"
            "5. Use the conversation history to resolve any pronouns or references "
            "   (e.g. 'he', 'it', 'that') to the correct entities from prior messages.\n"
            "6. You MUST always provide a non-empty response. If you truly cannot find any "
            "   relevant information, say something like: 'I don't have specific information "
            "   about this right now. This could be because the topic is very recent or not "
            "   covered in my available sources. You may want to check a live news source "
            "   for the latest updates.'\n"
            "7. NEVER return an empty or blank answer.\n\n"
            "Final Answer:"
        )

        try:
            final_msg = self.llm.invoke(final_prompt)
            answer = getattr(final_msg, "content", str(final_msg))
            # Append tool usage info
            if used_tool_name and used_tool_name != "none":
                answer += f"\n\n(Tool Used: {used_tool_name})"
        except Exception as e:
            answer = f"LLM error during final answer: {e}"

        state.answer = answer

        # Persist the assistant reply in messages for memory continuity
        state.messages.append(AIMessage(content=answer))

        return state
