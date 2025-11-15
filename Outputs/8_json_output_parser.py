from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser() # We cannot enforce the schema, LLM decides

template = PromptTemplate(
    template="Give me name, age and city of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()}, # This is filled before runtime
)

prompt = template.format()
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)
