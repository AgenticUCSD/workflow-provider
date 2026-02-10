import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# environment variables populated from .env file
# api_key, endpoint, etcs
load_dotenv()

model = ChatOpenAI(
    model="gpt-4.1",
)