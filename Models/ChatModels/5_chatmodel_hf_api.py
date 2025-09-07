from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

repo_id = "deepseek-ai/DeepSeek-R1-0528"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is Capital of India?")

print(response.content)
