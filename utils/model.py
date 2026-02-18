from langchain_openai import ChatOpenAI
from utils.config import OPENAI_API_KEY


model = ChatOpenAI(
    model="gpt-4.1",
    api_key=OPENAI_API_KEY,
)
