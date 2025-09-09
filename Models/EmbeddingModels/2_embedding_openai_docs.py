from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

docs = [
    "Delhi is capital of India",
    "Paris is capital of France",
    "London is capital of UK",
    "Berlin is capital of Germany",
]

result = embedding_model.embed_documents(docs)

print(str(result))
