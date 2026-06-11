import os 
from dotenv import load_dotenv
load_dotenv()
from  langchain_ollama import OllamaLLM,OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#langsmith tracking  
os.environ['LANGCHAIN_API_KEY']=os.getenv("langchain_api_key")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']=os.getenv("langchain_project")

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that will give all AL,ML,Genai related information to the user."),
        ("user","question: {question}")
    ]
)

import streamlit as st
st.title("Ollama App with GEMMA 4:31B-Cloud")
question = st.text_input("Ask your question here")
llm=OllamaLLM(model="gemma4:31b-cloud")
out_put_parser=StrOutputParser()
chain=prompt | llm | out_put_parser
if question:
    st.write(chain.invoke({"question": question}))