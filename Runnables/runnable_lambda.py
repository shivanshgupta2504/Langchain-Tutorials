from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# A small function to provide word count
# def count_words(text):
#     return len(text.split())

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

create_joke = PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=['topic']
)

joke_generation_chain = create_joke | model | parser

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda text: len(text.split())), # or 'word_count': RunnableLambda(word_count)
})

final_chain = joke_generation_chain | parallel_chain
res = final_chain.invoke({"topic": "AI"})
print(res)
