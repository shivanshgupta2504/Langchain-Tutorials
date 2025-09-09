from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about Virat Kohli"

doc_embeddings = embedding_model.embed_documents(documents)
query_embedding = embedding_model.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], doc_embeddings)[0] # Always pass 2D array, returns 2D array

most_similar_doc_index = np.argmax(similarity_scores)

print(f"Most similar document to the query: {documents[most_similar_doc_index]}")
