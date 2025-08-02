import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, SerpAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

# ==== Fetch keys from Streamlit secrets only ====
groq_api_key = st.secrets.get("GROQ_API_KEY")
serpapi_key = st.secrets.get("SERPAPI_KEY")

# Warnings if missing
if not groq_api_key:
    st.sidebar.warning("Groq API Key missing: set GROQ_API_KEY in Streamlit secrets.")
if not serpapi_key:
    st.sidebar.warning("SerpAPI Key missing: set SERPAPI_KEY in Streamlit secrets.")

# ==== Tool wrappers ====
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Only create search tool if key is available
if serpapi_key:
    search_wrapper = SerpAPIWrapper(serpapi_api_key=serpapi_key)
    search = Tool(
        name="search",
        description="Use this to search Google for current information and news. Always use lowercase 'search'",
        func=search_wrapper.run
    )
else:
    search = None

# ==== Streamlit UI ====
st.set_page_config(page_title="SearchSage", page_icon="🧠")
st.title("SearchSage: AI-Powered Knowledge Agent with Live Web, ArXiv & Wikipedia Search")
st.markdown(
    """
In this example, we're using `StreamlitCallbackHandler` to display the agent's reasoning and combine live web, ArXiv, and Wikipedia sources.
"""
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm SearchSage—ask me anything, and I'll fetch context from the web, ArXiv, and Wikipedia to answer."
        }
    ]

# Render existing chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input(placeholder="Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not groq_api_key:
        st.chat_message("assistant").write("❗ Groq API Key is missing. Please set GROQ_API_KEY in Streamlit secrets.")
    elif not serpapi_key:
        st.chat_message("assistant").write("❗ SerpAPI Key is missing. Please set SERPAPI_KEY in Streamlit secrets.")
    else:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", streaming=True)

        tools = []
        if search is not None:
            tools.append(search)
        tools.extend([arxiv, wiki])

        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handling_parsing_errors=True,
            verbose=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = search_agent.run(prompt, callbacks=[st_cb])
            except Exception as e:
                response = f"Error during agent run: {e}"
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write(response)
