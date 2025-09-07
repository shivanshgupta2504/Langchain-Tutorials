from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model_name="claude-3-7-sonnet-20250219")

result = model.invoke("What is the Capital of Russia?")

print(result.content)
