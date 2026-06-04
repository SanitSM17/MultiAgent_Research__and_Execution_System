import streamlit as st
from google import genai
from google.genai import types
from duckduckgo_search import DDGS
import chromadb
from chromadb.config import Settings
import os

# ==========================================
# 1. SETUP & INITIALIZATION
# ==========================================
st.set_page_config(page_title="Mega Agentic Workspace + Memory", page_icon="🤖", layout="wide")

st.title("🤖 Mega Agentic AI Workspace with Long-Term Memory")
st.caption("Powered by Gemini 2.5, DuckDuckGo, and ChromaDB — 100% Free Tier")

# Persistent directory for local vector memory
CHROMA_DATA_DIR = os.path.join(os.getcwd(), "chroma_memory")

# Initialize ChromaDB Client
@st.cache_resource
def get_vector_db():
    client = chromadb.PersistentClient(path=CHROMA_DATA_DIR)
    # Get or create a collection for agent memory
    collection = client.get_or_create_collection(name="agent_knowledge_base")
    return collection

chroma_collection = get_vector_db()

# Sidebar Config
with st.sidebar:
    st.header("🔑 Configuration")
    gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
    st.markdown("[Get a free Gemini API Key here](https://aistudio.google.com/)")
    
    st.markdown("---")
    st.header("🧠 Agent Knowledge Base (Memory)")
    uploaded_file = st.file_uploader("Upload background context (TXT file) to Agent Memory", type=["txt"])
    
    if uploaded_file and gemini_api_key:
        if st.button("Ingest into Memory"):
            with st.spinner("Embedding and memorizing document..."):
                try:
                    client = genai.Client(api_key=gemini_api_key)
                    file_content = uploaded_file.read().decode("utf-8")
                    
                    # Generate embedding using Gemini's free text embedding model
                    emb_response = client.models.embed_content(
                        model="text-embedding-004",
                        contents=file_content
                    )
                    embedding = emb_response.embeddings[0].values
                    
                    # Store in ChromaDB
                    chroma_collection.add(
                        embeddings=[embedding],
                        documents=[file_content],
                        ids=[uploaded_file.name]
                    )
                    st.success(f"Successfully memorized: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

# ==========================================
# 2. FREE AGENT TOOLS (Search & Vector Memory)
# ==========================================
def web_search_tool(query: str) -> str:
    """Searches the live web using DuckDuckGo for free."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if results:
                return "\n\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"Web search tool encountered an error: {str(e)}"
    return "No relevant web results found."

def vector_memory_tool(client, query: str) -> str:
    """Queries local ChromaDB memory for relevant historical contexts."""
    try:
        # Check if collection has documents
        if chroma_collection.count() == 0:
            return "No historical memory found in the local vector database."
        
        # Embed the query
        emb_response = client.models.embed_content(
            model="text-embedding-004",
            contents=query
        )
        query_embedding = emb_response.embeddings[0].values
        
        # Query database
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
        if results and results['documents'][0]:
            return f"Retrieved Context from Memory:\n{results['documents'][0][0]}"
    except Exception as e:
        return f"Memory retrieval failed: {str(e)}"
    return "No matching historical records found in memory."

# ==========================================
# 3. AGENT ORCHESTRATION CORE
# ==========================================
def call_agent(client, system_instruction: str, user_prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
        )
        return response.text
    except Exception as e:
        return f"Agent Error: {str(e)}"

# ==========================================
# 4. APP WORKFLOW EXECUTION
# ==========================================
user_topic = st.text_input("🎯 Enter a topic or business concept for the Agents to tackle:", 
                            placeholder="e.g., The rise of sustainable electric aviation")

if st.button("Launch Agentic Pipeline with Memory", type="primary"):
    if not gemini_api_key:
        st.error("Please provide a valid Gemini API key in the sidebar.")
    elif not user_topic:
        st.warning("Please enter a topic first.")
    else:
        try:
            client = genai.Client(api_key=gemini_api_key)
        except Exception as e:
            st.error(f"Failed to initialize Gemini Client: {e}")
            st.stop()

        col1, col2, col3 = st.columns(3)

        # ---- AGENT 1: The Smart Researcher (Web + Memory) ----
        with col1:
            st.subheader("🕵️‍♂️ 1. Memory-Augmented Researcher")
            with st.spinner("Scanning internal memories and real-world web data..."):
                # Run Tools
                live_web_data = web_search_tool(user_topic)
                internal_memory_data = vector_memory_tool(client, user_topic)
                
                researcher_instruction = (
                    "You are an elite Research Agent. Your job is to take raw web data "
                    "AND historical internal database records, cross-reference them, and build "
                    "a unified intelligence brief. Call out conflicting info if it exists."
                )
                
                research_prompt = (
                    f"Topic: {user_topic}\n\n"
                    f"--- LIVE WEB DATA ---\n{live_web_data}\n\n"
                    f"--- INTERNAL KNOWLEDGE BASE MEMORY ---\n{internal_memory_data}"
                )
                research_output = call_agent(client, researcher_instruction, research_prompt)
                
                st.markdown("### Final Intelligence Brief:")
                st.info(research_output)

        # ---- AGENT 2: The Critic / Strategist ----
        with col2:
            st.subheader("📊 2. Strategist Agent")
            with st.spinner("Evaluating risks and strategic opportunities..."):
                strategist_instruction = (
                    "You are a Senior Business Strategist. Review the provided intelligence brief. "
                    "Identify 3 major market opportunities and 2 severe risks/bottlenecks based on the research."
                )
                strategist_prompt = f"Review this Research Brief:\n{research_output}"
                strategy_output = call_agent(client, strategist_instruction, strategist_prompt)
                
                st.markdown("### SWOT & Opportunity Analysis:")
                st.warning(strategy_output)

        # ---- AGENT 3: The Content Creator / Executive ----
        with col3:
            st.subheader("✍️ 3. Executive Writer")
            with st.spinner("Synthesizing workflow into an executive report..."):
                writer_instruction = (
                    "You are a Chief Communications Officer. Take the Research Brief and the Strategy Analysis "
                    "and compile them into a beautifully formatted, comprehensive Executive Summary."
                )
                writer_prompt = f"Research:\n{research_output}\n\nStrategy:\n{strategy_output}"
                final_report = call_agent(client, writer_instruction, writer_prompt)
                
                st.markdown("### Final Generated Deliverable:")
                st.success(final_report)

        # Large Master Output Box
        st.markdown("---")
        st.header("📋 Final Master Output")
        st.text_area("Copy your comprehensive report:", value=final_report, height=400)