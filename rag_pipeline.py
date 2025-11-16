from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from cohere import Client as CohereClient
from langchain_cohere import CohereEmbeddings, ChatCohere
from pinecone import Pinecone
from langchain_pinecone import Pinecone as PineconeVectorStore
from dotenv import load_dotenv
import os
import pinecone
from fastapi import FastAPI
from pydantic import BaseModel
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware


##Loading Environment and Fast Api
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = None
co = None
vectorstore = None
bm25 = None

@app.on_event("startup")
def load_pipeline():
    global llm, co, vectorstore, bm25
    print("🟡 Starting pipeline setup...")
    ##Initializing LLMs and Rerank
    # llm = ChatOpenAI(model="gpt-4o-mini")
    try:
        llm = ChatCohere(model="command-a-03-2025")
        co = CohereClient(api_key=os.getenv("COHERE_API_KEY"))
        print("🟢 Cohere client + LLM initialized.")
    except Exception as e:
        print("❌ Error initializing Cohere/LLM:", e)

    ##Loading the document
    try:
        loader = PyPDFLoader("traffic_rules.pdf")
        document = loader.load()
        print("🟢 PDF loaded. Pages:", len(document))
    except Exception as e:
        print("❌ Error loading PDF:", e)
        return  # stop startup early

    ##Chunking
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(document)
        print("🟢 PDF split into chunks:", len(chunks))
    except Exception as e:
        print("❌ Error chunking PDF:", e)
        return


    ##Embedding
    try:
        embeddings = CohereEmbeddings(model="embed-english-v3.0")
        print("🟢 Cohere embeddings ready.")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("locallawgpt")
        print("🟢 Connected to Pinecone index: locallawgpt")
    except Exception as e:
        print("❌ Error connecting to Pinecone:", e)
        return

    

    ##Creating Retrivers
    try:
        vectorstore = PineconeVectorStore.from_documents(
            chunks,
            embeddings,
            index_name="locallawgpt"
        )
        print("🟢 Vectorstore created successfully.")
    except Exception as e:
        print("❌ Error creating vectorstore:", e)
        return

    try:
        bm25 = BM25Retriever.from_documents(chunks)
        print("🟢 BM25 retriever initialized.")
    except Exception as e:
        print("❌ Error initializing BM25 retriever:", e)
        return

    print("💚 Pipeline setup complete.")


##FastAPI
class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return "Welcome to LocalLaw GPT"

@app.post("/ask")
def ask_question(data: QueryRequest):
    global llm, co, vectorstore, bm25

    query = data.query

    # Combine retrievers
    vector_docs = vectorstore.similarity_search(query, k=20)
    bm25_docs = bm25._get_relevant_documents(query, run_manager=None)
    retrieved_docs = vector_docs + bm25_docs
    docs_text = [d.page_content for d in retrieved_docs]

    # Rerank
    rerank = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs_text,
        top_n=4
    )

    final_docs = [retrieved_docs[r.index] for r in rerank.results]

    # Summaries
    summarized_docs = []
    for chunk in final_docs:
        prompt = f"""
Summarize this text in simple, concise language, but don't miss important rules:

{chunk.page_content}
"""
        summary = llm.invoke(prompt).content
        summarized_docs.append(summary)

    # FIX: no backslashes inside the f-string
    context_text = "\n\n".join(summarized_docs)

    full_prompt = f"""
Use the following context to answer the user's question clearly and neatly.

Context:
{context_text}

Question:
{query}

Answer:
"""

    final_answer = llm.invoke(full_prompt).content

    return {"answer": final_answer}