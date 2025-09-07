from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=512,
    )
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the Capital of Finland?")
print(response.content)
