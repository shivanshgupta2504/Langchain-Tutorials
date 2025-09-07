from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4")
# Other parameters you can set:
# temperature=0.7 - Controls the randomness of the output.
# max_completion_tokens=256 - Limits the maximum number of tokens in the output.

result = model.invoke("What is the Capital of India?")
print(result.content)
