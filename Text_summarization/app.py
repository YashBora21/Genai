import streamlit as st
import validators

from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain

from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader
)

from langchain_ollama import ChatOllama


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")

st.subheader("🔗 Summarize URL")

st.markdown("""
Enter a **YouTube Video URL** or **Website URL**
and get a concise summary using LangChain + Ollama.
""")


# ---------------- LLM ----------------

llm = ChatOllama(
    model="gemma4:31b-cloud"
)


# ---------------- USER INPUT ----------------

url = st.text_input(
    "URL",
    placeholder="Enter YouTube or Website URL",
    label_visibility="collapsed"
)


# ---------------- PROMPT ----------------

prompt_template = """
Provide a concise summary of the following content
in about 500 words.

{text}

Summary:
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=prompt_template
)


# ---------------- BUTTON ----------------

if st.button("Summarize the Content From YT OR Website"):

    if not url.strip():

        st.error("Please provide a URL.")

    elif not validators.url(url):

        st.error("Please provide a valid URL.")

    else:

        try:

            with st.spinner("Loading and summarizing..."):

                # -------- YouTube --------

                if "youtube.com" in url or "youtu.be" in url:

                    try:

                        loader = YoutubeLoader.from_youtube_url(
                            url,
                            add_video_info=False,
                            language=["en"]
                        )

                        docs = loader.load()

                    except:

                        loader = YoutubeLoader.from_youtube_url(
                            url,
                            add_video_info=False,
                            language=["hi"]
                        )

                        docs = loader.load()

                # -------- Website --------

                else:

                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={
                            "User-Agent":
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/131.0 Safari/537.36"
                        }
                    )

                    docs = loader.load()

                # -------- Summarization --------

                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                result = chain.invoke(
                    {
                        "input_documents": docs
                    }
                )

                st.success("Summary Generated Successfully!")

                st.write(result["output_text"])

        except Exception as e:

            st.exception(e)