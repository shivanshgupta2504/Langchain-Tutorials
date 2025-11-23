from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

linkedin_post = PromptTemplate(
    template="Write a LinkedIn post on {topic}",
    input_variables=['topic']
)

twitter_post = PromptTemplate(
    template="Write a Twitter post on {topic}",
    input_variables=['topic']
)

chain = RunnableParallel({
    "linkedin_post": linkedin_post | model | parser,
    "twitter_post": twitter_post | model | parser,
})

result = chain.invoke({'topic': "Artificial General Intelligence(AGI)"})
print(result['linkedin_post'])
print(result['twitter_post'])
