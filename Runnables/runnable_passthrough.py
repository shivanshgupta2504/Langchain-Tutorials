from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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

joke_generation_chain = create_joke | model | parser

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explain_joke': explain_joke | model | parser,
})

chain = joke_generation_chain | parallel_chain

result = chain.invoke({'topic': "Cricket"})
print(result['joke'])
print(result['explain_joke'])
