# Requires: pip install langchain openai faiss-cpu tiktoken
# Set OPENAI_API_KEY in env before running.

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

# 1) Load your documents (example: simple list of strings)
raw_docs = [
    "LangChain is a framework for developing applications with LLMs.",
    "RAG stands for Retriever-Augmented Generation.",
    "FAISS is an efficient vector index for nearest-neighbor search."
]

# 2) Split into chunks (important for long docs)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = []
for i, d in enumerate(raw_docs):
    for c in splitter.split_text(d):
        chunks.append(Document(page_content=c, metadata={"source": f"doc-{i}"}))

# 3) Create embeddings and FAISS index
embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY
faiss_index = FAISS.from_documents(chunks, embeddings)

# 4) Create retriever and retrieval-based QA chain
retriever = faiss_index.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # or "gpt-4o", or "gpt-3.5-turbo"
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 5) Query
question = "What does RAG mean?"
answer = qa.run(question)
print("Answer:", answer)