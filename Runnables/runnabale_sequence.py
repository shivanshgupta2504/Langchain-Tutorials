from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

create_joke = PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=['topic']
)

explain_joke = PromptTemplate(
    template="Explain the given joke\n{joke}",
    input_variables=['joke']
)

chain = RunnableSequence(create_joke, model, parser, explain_joke, model, parser)
# This could also be written as chain = create_joke | model | parser | explain_joke | model | parser
print(chain.invoke({'topic': "AI"}))

