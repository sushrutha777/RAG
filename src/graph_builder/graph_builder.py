"""Graph builder for LangGraph workflow"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes

class GraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
        self.memory = MemorySaver()
    
    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        # Create state graph
        builder = StateGraph(RAGState)
        
        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("decision_engine", self.nodes.decision_engine)
        builder.add_node("responder", self.nodes.generate_answer)
        
        # Set entry point
        builder.set_entry_point("retriever")
        
        # Add edges
        builder.add_edge("retriever", "decision_engine")
        
        # Conditional routing from decision_engine
        # ALL paths go to responder — even "out_of_scope" — so the ReAct
        # agent can fall back to web_search / wikipedia for answers the
        # indexed documents don't cover.
        def route_decision(state: RAGState):
            # Always route to responder; it has web_search for out-of-scope
            # and time-sensitive queries.
            return "responder"
                
        builder.add_conditional_edges(
            "decision_engine",
            route_decision,
            {
                "responder": "responder"
            }
        )
        
        builder.add_edge("responder", END)
        
        # Compile graph with memory checkpointer
        self.graph = builder.compile(checkpointer=self.memory)
        return self.graph
    
    def run(self, question: str, thread_id: str = None, chat_history: list = None) -> dict:
        """
        Run the RAG workflow
        
        Args:
            question: User question
            thread_id: Optional thread ID for conversation memory.
            chat_history: List of prior chat dicts [{"role": "user"|"assistant", "content": str}].
                          These are converted to LangChain message objects so the agent
                          can resolve pronouns and follow-up references.
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        
        # Build full message history from Streamlit chat_history
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        
        # Only append the current question if it's not already the last message
        if not messages or getattr(messages[-1], "content", None) != question:
            messages.append(HumanMessage(content=question))
        
        initial_state = RAGState(
            question=question,
            messages=messages
        )
        
        # Build config with thread_id for memory persistence
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        result = self.graph.invoke(initial_state, config=config)
        
        # Ensure result is a dict for app.py
        if not isinstance(result, dict):
            # Try Pydantic v2
            if hasattr(result, "model_dump"):
                return result.model_dump()
            # Try Pydantic v1
            if hasattr(result, "dict"):
                return result.dict()
                
        return result