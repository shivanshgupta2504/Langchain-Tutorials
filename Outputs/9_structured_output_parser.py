from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(model="deepseek-r1:1.5b")

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about the {topic} \n {format_instructions}",
    input_variables=['topic'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({'topic': 'Black Hole'})
result = model.invoke(prompt)
print(result.content)
final_result = parser.parse(result.content)
print(final_result)
