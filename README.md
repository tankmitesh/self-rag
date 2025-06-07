# SELF-RAG

**SELF-RAG** is a modular Retrieval-Augmented Generation (RAG) system designed to support customizable indexing, chunking, retrieval, and response generation using Python and LangChain.

> Compatible with **Python 3.10.16**

---

## 📦 Features

- **Chunk Relevance Evaluation**  
  Automatically assess and filter chunks based on relevance to the query.

- **Index Creation**  
  Build and store vector indices from your documents.

- **LangChain-Based Document Splitting**  
  Use LangChain to efficiently split large documents into manageable chunks.

- **Dynamic Retriever Selection**  
  Choose appropriate retrievers based on the query using custom logic.

- **Response Generation**  
  Generate answers using LLMs based on retrieved context.

---

## 📁 Project Structure

```

self-rag/
├── chunk\_relevance\_evaluator.py   # Relevance scoring for document chunks
├── create\_index.py                # Index creation script
├── langchain\_splitter.py         # Document splitting using LangChain
├── main.py                        # Entry point for RAG pipeline
├── response\_generator.py         # Generates final responses from LLM
├── retriver\_decider.py           # Decides which retriever to use
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/tankmitesh/self-rag.git
cd self-rag
````

### 2. Create a Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> Make sure your `requirements.txt` includes dependencies like `langchain`, `openai`, `pinecone-client` and `python-dotenv`.

### 4. Set Up Environment Variables

Create a `.env` file in the root directory with the following content:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key

# Pinecone API Key and Environment
PINECONE_API_KEY=your_pinecone_api_key
```

> Ensure `.env` is listed in `.gitignore` to avoid accidentally committing sensitive information.

---

## 🧠 Usage

### Step 1: Index Documents

```bash
python create_index.py
```

This will:

* Load documents
* Split them using `langchain_splitter.py`
* Create embeddings
* Store them in Pinecone

### Step 2: Run the Main Pipeline

```bash
python main.py
```

This will:

1. Split the query if needed
2. Evaluate chunk relevance
3. Choose the best retriever
4. Generate a response using retrieved data and LLM

---

## 📜 License

This project is licensed under the terms of the LICENSE file.

