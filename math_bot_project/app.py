import streamlit as st
from langchain_classic.chains import LLMChain,LLMMathChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool,initialize_agent
from dotenv import load_dotenv
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_ollama import ChatOllama
llm=ChatOllama(model="gemma4:31b-cloud")

st.set_page_config(page_title="Text to Math Problem Solver and Data Search assistant",page_icon='|_|')
st.title("Text to Math Problem Solver and Data Search assistant")

##tools

Wiki=WikipediaAPIWrapper()
wiki_tool=Tool(
    name="Wikipedia",
    func=Wiki.run,
    description="tool for seraching the internet to find the various information on the topics mentions"
)

##intializes the math tool

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="the tool should provide the math related problem solution in descriptive and interactive and easy way understand by user. Only input mathematical expression need to provided "
)

prompt = """
You are an expert Mathematics Tutor.

For every mathematics question:

1. Identify the topic.
2. Solve it step by step.
3. Explain each step clearly.
4. Use mathematical notation and LaTeX.
5. Show the final answer separately.

Question:
{Question}

Answer:
"""
template=PromptTemplate(input_variables=["Question"],template=prompt)

#math chain

chain=LLMChain(llm=llm,prompt=template)

reasoning_tool = Tool(
    name="reasoning_tool",
    func=chain.run,
    description="""
    Useful for:
    - Integration
    - Differentiation
    - Limits
    - Probability
    - Statistics
    - Algebra
    - Calculus
    - Step by step mathematical explanations
"""
)
agent=initialize_agent(tools=[wiki_tool,calculator,reasoning_tool],llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,handle_parsing_errors=True,max_iterations=15,early_stopping_method="generate")
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I am a Math chatBot Who can answer all Math Answer"}

    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

##funstion to genrate the function



## interaction
question=st.text_area(label="Enter Your Question here")
if st.button("find my answer"):
    if question:
        with st.spinner("Genrating the response...."):
            st.session_state.messages.append({"role":'user',"content":question})
            st.chat_message("user").write(question)
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=agent.invoke(    {"input": question},callbacks=[st_cb])
            answer = response["output"]
            st.session_state.messages.append({'role':'assistant','content':answer})
            st.write('##  Response')
            st.markdown(response['output'])

    else:
        st.warning("Please enter the Question ")


