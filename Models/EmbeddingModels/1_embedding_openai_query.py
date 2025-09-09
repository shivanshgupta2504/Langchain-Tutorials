from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
# Higher dimension captures more semantic details but requires more storage and computation.
# Lower dimension is faster and uses less storage but may miss some nuances.

result = embedding_model.embed_query("Delhi is Capital of India")
print(str(result))
