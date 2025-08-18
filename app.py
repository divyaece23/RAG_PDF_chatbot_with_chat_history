### streamlit run sample_rag_app.py
# .venv\Scripts\activate

import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT']="Q&A Chatbot"
os.environ['LANGCHAIN_TRACING_V2']="true"
groq_api_key=os.getenv("GROQ_API_KEY")

def create_vector_embeddings(uploaded_files):
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    embeddings_hug = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_hug)
    return vectorstore

def get_session_history(session:str)->BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    return  st.session_state.store[session_id]

## can also use Gemma-7b-It mnodel
llm=ChatGroq(model="Llama3-8b-8192",groq_api_key=groq_api_key)

st.title("Q&A Chatbot about Research Papers")
st.write('Upload PDF file')

session_id = st.text_input('Session_id', value='default_seesion')
if 'store' not in st.session_state:
    st.session_state.store={}

uploaded_files=st.file_uploader('Choose PDF file', type='pdf', accept_multiple_files= True)

if uploaded_files:
    vectorstore = create_vector_embeddings(uploaded_files)
    retriever = vectorstore.as_retriever()    

    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    # Answer question
    system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    user_input = st.text_input("Your question:")
    if user_input:
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id":session_id}
            },  
        )
        st.write("Assistant:", response['answer'])
else:
    st.write('Please upload the pdf file')
