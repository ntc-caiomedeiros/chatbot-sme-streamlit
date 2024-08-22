import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

os.environ["CHROMA_PATH"] = "C:/Users/User/PROJETO_LLM/chroma/"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def get_embedding_function():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return embeddings

client = chromadb.PersistentClient(
    path=os.environ['CHROMA_PATH'],
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

db = Chroma(
    client=client, collection_name="chatbot-sme",
    persist_directory=os.environ['CHROMA_PATH'], embedding_function=get_embedding_function()
)

system_message = """
You are a IT Helpdesk Support Assistant that attends all the schools of Rio de Janeiro, helping them to solve issues with
the educational softwares.
The educational softwares are Web Enable, Sisbens, SIGMA, Processo Rio, Fincon, Ergon, DESESC and Educa Censo.
You have to answer in a simple and non-technical language, considering you are helping teachers and school campus inspectors.
If the user describes the problem in a vague way, ask it for more details.

Give the solution with detailed steps, including instructions of OS navigation and operation like creating shortcuts, changing folder
and file permissions, changing environment variables, etc.

If the solution steps include downloading softwares that are required by the educational softwares, you must give the instalation instructions
for these required softwares. You must quote the reference links of the installation instructions from the official software website.

At yor first answer, you must not tell the user to search for any other technical support such as iplan rio. If the user needs more help, it should ask you.

Your answer language must be Brazilian Portuguese.    
    """

messages = [
    SystemMessage(content=system_message)
]

def augment_prompt(question:str):
    # get top 3 results from knowledge base
    docs = db.similarity_search(question, k=3)
    context = "\n".join([x.page_content for x in docs])
    # get the text from the results
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {context}

    Query: {question}"""
    return augmented_prompt

chat = ChatOpenAI(model="gpt-4o-mini")


st.title("ChatBot SME")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
        elif message["role"] == "user":
            messages.append(HumanMessage(content=augment_prompt(message["content"])))

prompt = st.chat_input("Como posso te ajudar?")
if prompt:
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user", "content": prompt})
    messages.append(HumanMessage(content=augment_prompt(prompt)))

    with st.chat_message("assistant"):
        full_response = ""
        message_placeholder = st.empty()
        stream = chat.stream(messages)
        for chunk in stream:
            full_response += chunk.content
            message_placeholder.markdown(full_response + "_")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role":"assistant", "content":full_response})
    messages.append(AIMessage(content=full_response))