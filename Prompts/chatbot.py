from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
from typing import List

load_dotenv()

model = ChatOpenAI()

chat_history: List[BaseMessage] = [
    SystemMessage(content="You are a helpful AI assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        print("Thank you for using the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"AI: {result.content}")

print(chat_history)
