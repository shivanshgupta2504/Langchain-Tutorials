from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

report_template = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic'],
)

summary_template = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=['text'],
)

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

chain = report_template | model | parser | summary_template | model | parser

res = chain.invoke({'topic': "String Theory"})

print(res)

