from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage

# chat_template = ChatPromptTemplate([
#     SystemMessage(content='You are a helpful {domain} expert.'),
#     HumanMessage(content="Explain in simple terms, what is {topic}?")
# ])

# Another (preferred) way to create chat prompt template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', "Explain in simple terms, what is {topic}?")
])

prompt = chat_template.invoke({'domain': 'Cricket', 'topic': 'LBW'})

print(prompt)
