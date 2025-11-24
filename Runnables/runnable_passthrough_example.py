# This is just a example code to show the working of RunnablePassthrough
from langchain.schema.runnable import RunnablePassthrough

passthrough = RunnablePassthrough()

print(passthrough.invoke(2)) # What you input, returns the same
print(passthrough.invoke({"name": "Shivansh"}))
