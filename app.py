import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, SerpAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()

# ==== Secrets only ====
groq_api_key = st.secrets.get("GROQ_API_KEY")
serpapi_key = st.secrets.get("SERPAPI_KEY")

if not groq_api_key:
    st.sidebar.warning("Groq API Key missing: set GROQ_API_KEY in Streamlit secrets.")
if not serpapi_key:
    st.sidebar.warning("SerpAPI Key missing: set SERPAPI_KEY in Streamlit secrets.")

# ==== Tool wrappers ====
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = None
if serpapi_key:
    search_wrapper = SerpAPIWrapper(serpapi_api_key=serpapi_key)
    search = Tool(
        name="search",
        description="Use this to search Google for current information and news. Always use lowercase 'search'",
        func=search_wrapper.run
    )

# ==== Streamlit UI ====
st.set_page_config(page_title="SearchSage", page_icon="🧠")
st.title("SearchSage: AI-Powered Knowledge Agent with Live Web, ArXiv & Wikipedia Search")
st.markdown(
    """
A history-aware assistant: follow-up questions get reformulated in context before search and retrieval.
"""
)

# Initialize message history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of dicts: {"role":..., "content":...}

# Display existing history
for msg in st.session_state["chat_history"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Input box
if user_query := st.chat_input(placeholder="Ask a question..."):
    # Append user message
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Preconditions
    if not groq_api_key:
        st.chat_message("assistant").write("❗ Groq API Key is missing. Please set GROQ_API_KEY in Streamlit secrets.")
    elif not serpapi_key:
        st.chat_message("assistant").write("❗ SerpAPI Key is missing. Please set SERPAPI_KEY in Streamlit secrets.")
    else:
        # 1. Reformulate follow-up question using chat history
        # Build prompt: system + history + latest question -> standalone question
        reformulate_system = SystemMessagePromptTemplate.from_template(
            "Given the conversation history and the latest user question, "
            "rewrite the latest question so that it is a self-contained standalone question. "
            "If it's already standalone, just output it unchanged. Keep it concise."
        )
        # Prepare history string
        history_messages = []
        for m in st.session_state["chat_history"][:-1]:  # all except latest
            role = m["role"]
            content = m["content"]
            if role == "user":
                history_messages.append(f"User: {content}")
            else:
                history_messages.append(f"Assistant: {content}")
        history_block = "\n".join(history_messages) if history_messages else "No prior conversation."

        reformulate_prompt = ChatPromptTemplate.from_messages([
            ("system", reformulate_system.prompt.template),
            ("user", "Conversation history:\n" + history_block + "\n\nLatest question:\n" + user_query + "\n\nStandalone version:")
        ])

        reformulator_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192", streaming=False)
        reformulation_chain = LLMChain(llm=reformulator_llm, prompt=reformulate_prompt)
        try:
            standalone_question = reformulation_chain.run()
            standalone_question = standalone_question.strip().strip('"')
        except Exception:
            standalone_question = user_query  # fallback

        # 2. Prepare agent with tools
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

        # 3. Run agent on reformulated question
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = search_agent.run(standalone_question, callbacks=[st_cb])
            except Exception as e:
                response = f"Error during agent run: {e}"
            # Append assistant message
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            st.write(response)



