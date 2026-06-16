import streamlit as st
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_classic.agents import initialize_agent,AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_ollama  import ChatOllama
from dotenv import load_dotenv
import os
load_dotenv()
from langchain.tools import tool

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
ollama_api=os.getenv("Olla")
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=5,doc_content_chars_max=250)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=5,doc_content_chars_max=250)
arxiv_tool=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
search=DuckDuckGoSearchRun(name="search")
@tool
def document_search(query: str) -> str:
    """
    Search uploaded PDF documents.
    """

    if "retriever" not in st.session_state:
        return "No PDF uploaded."

    docs = st.session_state.retriever.invoke(query)

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )
st.title(" Lancahin -chat messages")
uploaded_pdf = st.sidebar.file_uploader(
    "Upload Research Paper / PDF",
    type=["pdf"]
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {
            "role":"Assistant",
            "content":"hi, i am a chat bot who search web. How can help you"

        }

    ]
agent_kwargs = {
    "prefix": """
You are a helpful research assistant.

Tools available:

1. document_search
   - Use for uploaded PDFs and research papers.

2. wikipedia
   - Use for historical facts and people.

3. search
   - Use for latest web information.

If user asks about uploaded PDF,
ALWAYS use document_search first.
"""
}
for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])
if uploaded_pdf and "retriever" not in st.session_state:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.from_documents(
        documents,
        embeddings
    )

    st.session_state.retriever = vectordb.as_retriever(
        search_kwargs={"k": 4}
    )

    st.sidebar.success("PDF Indexed Successfully")
if prompt:=st.chat_input(placeholder="what is Machine Learning"):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").write(prompt)


        llm = ChatOllama(
            model="gemma4:31b-cloud"
        )
        tools=[search,document_search,wiki_tool]
        search_agent=initialize_agent(tools,llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,agent_kwargs=agent_kwargs,verbose=True)
        with st.chat_message("assistant"):
               st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
               response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
               st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )
               st.write(response)

