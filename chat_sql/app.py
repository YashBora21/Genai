import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit 
from langchain_community.utilities.sql_database import SQLDatabase 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.callbacks import StreamingStdOutCallbackHandler
from langchain_classic.agents import create_sql_agent
from sqlalchemy import create_engine
import sqlite3
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from sqlalchemy.engine import URL

class StreamHandler(BaseCallbackHandler):

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


st.set_page_config(page_title="Langchain: Chat With SQL DB ",page_icon="$")
st.title("Langchain: Chat With SQL DB ")
INJECTION_WARNING = """
⚠️ SQL agents can be vulnerable to prompt injection attacks.

• Use a read-only database user.
• Restrict permissions to SELECT only.
• Never use the root user in production.
• Validate generated SQL before execution.
• Avoid exposing API keys, passwords, or environment variables.

Read more:
https://python.langchain.com/docs/security/
"""
st.warning(INJECTION_WARNING)

LOCAL_DB="USE_LOCALDB"
MYSQL="USE_MYSQL"

radio_opt=["Use sql3lite student db","connect your own sql db"]
selected_opt=st.sidebar.radio(label="choose the DB Which you want to chat",options=radio_opt)
if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host=st.sidebar.text_input("Enter The MYSQL Host")
    mysql_user=st.sidebar.text_input("Enter The MYSQL User")
    mysql_password=st.sidebar.text_input("Enter The MYSQL Password",type='password')
    mysql_database=st.sidebar.text_input("Enter The MYSQL Database")
else:
    db_uri=LOCAL_DB

api_key=st.sidebar.text_input(label="Enter Groq API Key",type="password")
if not db_uri :
    st.info("Please Enter the Database URI")

if not api_key :
    st.info("Please Enter the GROQ API KEY")
    st.stop()

##llm
llm=ChatGoogleGenerativeAI(google_api_key=api_key,model="gemini-2.5-flash")

@st.cache_resource(ttl='2h')
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_database=None):
    if db_uri==LOCAL_DB:
        dbfilepath=(Path(__file__).parent/"studentdb").absolute()
        creator=lambda:sqlite3.connect(f"file:{dbfilepath}?mode=ro",uri=True)
        return SQLDatabase(create_engine("sqlite:///",creator=creator))
    elif db_uri==MYSQL:
        if not(mysql_host and mysql_user and mysql_password and mysql_database):
            st.error("please Provide all Information")
            st.stop()
        
        connection_url = URL.create(
                drivername="mysql+pymysql",
                username=mysql_user,
                password=mysql_password,
                host=mysql_host,
                port=3306,
                database=mysql_database
        )

        engine = create_engine(connection_url)

        return SQLDatabase(engine)

        
db=""
if db_uri==MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_database)
else :
    db=configure_db(db_uri)

##toolkit
toolkit=SQLDatabaseToolkit(db=db,llm=llm)
agent=create_sql_agent(llm=llm,toolkit=toolkit,verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

if "messages" not in st.session_state or st.sidebar.button("Clear Message History"):
    st.session_state["messages"]=[{"role":"assistant","content":"How Can I Help YOu?"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

user_query=st.chat_input(placeholder="Ask Any Thing From Database")
if user_query:
    st.session_state.messages.append({"role":"user","content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        stream_container = st.empty()

        call = StreamHandler(stream_container)

        response = agent.run(
            user_query,
            callbacks=[call]
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

