from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

file_path = r"C:\Users\Shivansh Gupta\OneDrive\Desktop\Data Science\My Data Science\Langchain-Tutorials\RAG\Document_Loaders\dl-curriculum.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator="",
)

result = splitter.split_documents(docs)

print(result[0])
