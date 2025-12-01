from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

template = PromptTemplate(
    template="Write a summary of following poem\n{poem}",
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader(file_path="cricket.txt", encoding="utf-8")

docs = loader.load()

chain = template | model | parser
response = chain.invoke({'poem': docs[0].page_content})
print(response)

