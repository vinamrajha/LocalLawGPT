
# LocalLawGPT 🚦  
A FastAPI-based RAG system that answers questions about **Indian traffic rules** using:

- **Cohere LLM (Command model)**
- **Cohere Embeddings**
- **Cohere Reranker**
- **Pinecone Vector Database**
- **BM25 Hybrid Retrieval**
- **PDF-based document ingestion**
- **Fully deployed on Render**

---

## 🚀 Features

### 🔍 Retrieval-Augmented Generation (RAG)
- Combines **vector search (Pinecone)** and **BM25 keyword search** for maximum accuracy.
- Reranks results using **Cohere's Rerank v3**.
- Summarizes retrieved chunks using **Cohere LLM**.
- Produces final answers grounded in authoritative traffic rules.

---

## 📁 Project Structure

```
LocalLawGPT/
│
├── rag_pipeline.py       # Main FastAPI backend + RAG pipeline
├── traffic_rules.pdf     # Source document for chunking
├── requirements.txt      # Dependencies for deployment
├── render.yaml           # Render deployment config
└── .env                  # Environment variables (local only, not committed)
```

---

## ⚙️ Technologies Used

### 🧠 LLM & Embeddings  
- **Cohere Command-A (2025)**
- **Cohere English Embeddings v3**

### 🔎 Retrieval  
- **Pinecone Vectorstore**
- **BM25 Retriever**

### 🏗 Frameworks  
- **FastAPI**
- **Uvicorn**
- **LangChain**

### ☁️ Deployment Platform  
- **Render Web Services**

---

## 🔧 How It Works

1. On server startup (`startup` event):
   - Loads PDF  
   - Splits into text chunks  
   - Creates BM25 index  
   - Embeds chunks using Cohere  
   - Pushes embeddings to Pinecone  
   - Prepares RAG pipeline  

2. On user query:
   - Hybrid search → (Pinecone + BM25)  
   - Reranker picks top chunks  
   - Each chunk summarized  
   - Final LLM answer generated with context  

---

## ▶️ API Endpoints

### **GET /**
Returns a welcome message.

### **POST /ask**

#### Request Body:
```json
{
  "query": "What are the speed limits on Indian highways?"
}
```

#### Response:
```json
{
  "answer": "..."
}
```

---

## 🔐 Environment Variables

Create a `.env` file:

```
COHERE_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_ENV=your_environment
```

---

## 🚀 Deployment (Render)

1. Push project to GitHub  
2. Create Render Web Service  
3. Render auto-detects `render.yaml`  
4. Add environment variables  
5. Deploy  

Start command inside `render.yaml`:
```
uvicorn rag_pipeline:app --host 0.0.0.0 --port 10000
```

---

## 📝 Notes
- Do **not** commit `.env` to GitHub.
- Make sure Pinecone index `locallawgpt` exists before deployment.
- Logs help diagnose startup issues (`print` checkpoints included).

---

## 👨‍💻 Author
**Vinamra Kumar**  
Creator of LocalLawGPT — an intelligent assistant for Indian traffic rules.

---

## ⭐ If you want to extend
- Add authentication  
- Add chat streaming  
- Add a front-end UI  
- Optimize chunking and retrieval  
- Include more legal PDFs  

---

Enjoy building smarter AI tools! 🚀
