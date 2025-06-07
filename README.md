Here’s a full `README.md` tailored for your `self-rag` project, assuming it runs on **Python 3.10.16** and based on the files visible in your project directory:

---

```markdown
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
├── chunk_relevance_evaluator.py   # Relevance scoring for document chunks
├── create_index.py                # Index creation script
├── langchain_splitter.py         # Document splitting using LangChain
├── main.py                        # Entry point for RAG pipeline
├── response_generator.py         # Generates final responses from LLM
├── retriver_decider.py           # Decides which retriever to use
├── .gitignore
├── LICENSE
├── README.md
├── requierements.txt

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

> If `requirements.txt` does not exist yet, create it with the packages used in the project, like `langchain`, `openai`, `faiss-cpu`, etc.

### 4. Set Up Environment Variables

Create a `.env` file with the following:

```env
OPENAI_API_KEY=your_openai_key
```

---

## 🧠 Usage

### Index Documents

```bash
python create_index.py
```

### Run the Main Pipeline

```bash
python main.py
```

This will:

1. Split documents
2. Evaluate chunk relevance
3. Retrieve relevant info
4. Generate a response

---

## 📜 License

This project is licensed under the terms of the LICENSE file.
