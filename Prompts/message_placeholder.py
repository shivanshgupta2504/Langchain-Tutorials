from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# This MesssagePlace holder works for the chat history in the chatbots

# Chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# Load chat history
chat_history = []
with open("chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())

# print(chat_history)

# Create the prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': "Where is my refund?"})
print(prompt)
