from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

model = ChatOllama(model="deepseek-r1:1.5b")

class Person(BaseModel):
    name: str = Field(description="Name of person")
    age: int = Field(description="Age of person", gt=18)
    city: str = Field(description="City of person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of fictional {place} person \n {format_instructions}",
    input_variables=['place'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
final_res = chain.invoke({'place': 'australia'})

# prompt = template.invoke({'place': 'indian'})
#
# result = model.invoke(prompt)
#
# final_res = parser.parse(result.content)
print(final_res)
