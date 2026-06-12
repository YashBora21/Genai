import os
from dotenv import load_dotenv
load_dotenv()
from operator import itemgetter
import streamlit as st

from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)

from langchain_core.output_parsers import (
    StrOutputParser,
)

from langchain_core.chat_history import (
    BaseChatMessageHistory,
)

from langchain_community.chat_message_histories import (
    ChatMessageHistory,
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)

# =====================================================
# PAGE
# =====================================================

st.set_page_config(page_title="Conversational PDF RAG")

st.title("📄 Conversational RAG with PDF Upload")
st.write("Upload PDF files and chat with them.")

# =====================================================
# SESSION STORE
# =====================================================

if "store" not in st.session_state:
    st.session_state.store = {}

# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

# =====================================================
# EMBEDDINGS
# =====================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True,
)

# =====================================================
# VECTOR DB CREATION
# =====================================================

if uploaded_files:

    if "vectordb" not in st.session_state:

        documents = []

        with st.spinner("Processing PDFs..."):

            for uploaded_file in uploaded_files:

                temp_pdf = f"./{uploaded_file.name}"

                with open(temp_pdf, "wb") as file:
                    file.write(uploaded_file.getvalue())

                loader = PyPDFLoader(temp_pdf)

                docs = loader.load()

                documents.extend(docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )

            final_docs = splitter.split_documents(
                documents
            )

            st.write(
                f"Chunks Created: {len(final_docs)}"
            )

            vectordb = FAISS.from_documents(
                final_docs,
                embeddings,
            )

            st.session_state.vectordb = vectordb

            st.success(
                "Embeddings Created Successfully!"
            )

    # =================================================
    # RETRIEVER
    # =================================================

    retriever = (
        st.session_state.vectordb.as_retriever(
            search_kwargs={"k": 4}
        )
    )

    # =================================================
    # PROMPT
    # =================================================

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant.

                Answer the user's question using ONLY
                the retrieved context.

                Context:
                {context}

                If the answer is not available
                in the context, reply:

                I don't know.
                """
            ),
            MessagesPlaceholder(
                variable_name="history"
            ),
            (
                "human",
                "{question}"
            ),
        ]
    )

    # =================================================
    # FORMAT DOCS
    # =================================================

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    # =================================================
    # RAG CHAIN
    # =================================================

    rag_chain = (
        {
            "context":itemgetter("question")
                | retriever
                | RunnableLambda(format_docs),

            "question":
                RunnablePassthrough(),

            "history":
                lambda x: x.get(
                    "history",
                    []
                ),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # =================================================
    # CHAT HISTORY FUNCTION
    # =================================================

    def get_session_history(
        session_id: str,
    ) -> BaseChatMessageHistory:

        if (
            session_id
            not in st.session_state.store
        ):

            st.session_state.store[
                session_id
            ] = ChatMessageHistory()

        return st.session_state.store[
            session_id
        ]

    # =================================================
    # CONVERSATIONAL CHAIN
    # =================================================

    conversational_rag_chain = (
        RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )
    )

    # =================================================
    # SESSION ID
    # =================================================

    session_id = st.text_input(
        "Session ID",
        value="default_user",
    )

    # =================================================
    # USER INPUT
    # =================================================

    user_question = st.chat_input(
        "Ask a question about your PDFs..."
    )

    # =================================================
    # ASK
    # =================================================

    if user_question:

        with st.spinner("Thinking..."):

            response = (
                conversational_rag_chain.invoke(
                    {
                        "question":
                        user_question
                    },
                    config={
                        "configurable": {
                            "session_id":
                            session_id
                        }
                    },
                )
            )

        st.subheader("Answer")
        st.write(response)