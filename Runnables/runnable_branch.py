from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough, RunnableBranch, RunnableSequence

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

report_prompt = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

summary_prompt = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

report_generation_chain = RunnableSequence(report_prompt, model, parser)

branch_chain = RunnableBranch(
    (lambda text: len(text.split()) > 300, RunnableSequence(summary_prompt, model, parser)),
    RunnablePassthrough()
)

final_chain = report_generation_chain | branch_chain

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))

