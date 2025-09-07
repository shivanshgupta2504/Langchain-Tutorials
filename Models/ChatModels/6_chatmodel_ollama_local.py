from langchain_ollama import ChatOllama

model = ChatOllama(model="llama2:latest")

result = model.invoke("What is Capital of Greece?")

print(result.content)
