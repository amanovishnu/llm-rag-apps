from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from decouple import AutoConfig
from langserve import add_routes


config = AutoConfig(search_path='../notes-rag/')
GROQ_API_KEY = config("GROQ_API_KEY")
model = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY)
output_parser = StrOutputParser()

system_template = "translate the following into {language}."
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{input}')
])

chain = prompt_template | model | output_parser

# App Definition
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple api server using langchain runnable interfaces"
)

add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)