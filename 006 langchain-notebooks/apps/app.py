from decouple import AutoConfig
from icecream import ic
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


config = AutoConfig(search_path='../notes-rag/')

HF_TOKEN = config("HF_TOKEN")
LANGCHAIN_TRACING_V2 = config("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = config("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = config("LANGCHAIN_PROJECT")
lANGCHAIN_API_KEY = config("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant, please respond to the question asked ?"),
    ("user", "question: {question}")
])
llm = OllamaLLM(model="llama3.2")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

st.title("This is My Langchain demo with LLAMA3.2 model")
input_text = st.text_input("What Question do you have in Mind ?")

if input_text:
    st.write(chain.invoke({"question": input_text}))
