from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFacePipeline.from_model_id(
#     model_id="google/gemma-2b-it",
#     task="text-generation",
# )
#
# model = ChatHuggingFace(llm=llm)
model = ChatOpenAI()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on following text.\n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': "Black Hole"})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})
result = model.invoke(prompt2)
print(result.content)

