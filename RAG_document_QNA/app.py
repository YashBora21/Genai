import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# -------------------------------
# GROQ LLM
# -------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------------
# PROMPT
# -------------------------------

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

If the answer is not found in the context,
say "I don't know".

Context:
{context}

Question:
{question}
""")

# -------------------------------
# VECTOR DB CREATION
# -------------------------------

def create_vector_embeddings():

    if "vectordb" not in st.session_state:

        with st.spinner("Loading PDFs..."):

            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://127.0.0.1:11434"
            )

            loader = PyPDFDirectoryLoader(
                r"C:\Users\Lenovo\OneDrive\Desktop\Lanchain_k\RAG_document_QNA\report"
            )

            docs = loader.load()

            st.write(f"PDF Pages Loaded: {len(docs)}")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            final_docs = splitter.split_documents(docs)

            st.write(f"Chunks Created: {len(final_docs)}")

            # Debug Test
            sample_embedding = embeddings.embed_query(
                final_docs[0].page_content[:500]
            )

            st.write(
                f"Embedding Dimension: {len(sample_embedding)}"
            )

            vectordb = FAISS.from_documents(
                final_docs,
                embeddings
            )

            st.session_state.vectordb = vectordb

            st.success("Embeddings Created Successfully!")

# -------------------------------
# STREAMLIT UI
# -------------------------------

st.title("📄 Document Q&A using RAG")

question = st.text_input(
    "Enter your question"
)

if st.button("Create Document Embeddings"):
    try:
        create_vector_embeddings()
    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# QUERY
# -------------------------------

if question:

    if "vectordb" not in st.session_state:
        st.warning(
            "Please create embeddings first."
        )
        st.stop()

    retriever = st.session_state.vectordb.as_retriever(
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    with st.spinner("Searching..."):

        response = rag_chain.invoke(question)

    st.subheader("Answer")

    st.write(response.content)