import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever


class EmbeddingManager:
    def __init__(self, documents : list, index_name : str, embedding_size : int = 1536):

        load_dotenv()
        self.pinecone_api_key = os.getenv("PINECONE_API")
        self.documents = documents
        self.index_name = index_name
        self.embedding_size = embedding_size

    def create_database_index(self):

        pc = Pinecone(api_key = self.pinecone_api_key)

        if not pc.has_index(self.index_name):
            pc.create_index(name = self.index_name,
                            metric = "dotproduct",
                            dimension = self.embedding_size,
                            spec = ServerlessSpec(cloud = "aws", region = "us-east-1"))

        pc = pc.Index(self.index_name)

        return pc

    def sparse_embeddings(self):

        # Set up sparse embeddings using BM25
        bm25 = BM25Encoder().default()

        # Fit the BM25 model to the documents
        bm25.fit(self.documents)

        # Store the BM25 model to a file
        bm25.dump("bm25_encoder.json")

        # Load the BM25 model from the file
        bm25_encoder = BM25Encoder().load("bm25_encoder.json")

        return bm25_encoder
 
    def dense_embeddings(self, model = "text-embedding-3-small") -> OpenAIEmbeddings:
        # Set up dense embeddings using OpenAI
        return OpenAIEmbeddings(model = model)
    

    def get_retriever(self):

        pincon_db = self.create_database_index()
        dense_encoder = self.dense_embeddings()
        bm25_encoder = self.sparse_embeddings()
    
        # Create a hybrid retriever using both sparse and dense embeddings
        retriever = PineconeHybridSearchRetriever(embeddings = dense_encoder,
                                                  sparse_encoder = bm25_encoder,
                                                  index = pincon_db)
        

        retriever.add_texts(self.documents)
        
        return retriever