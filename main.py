import os 
from dotenv import load_dotenv
import numpy as np
import streamlit as st
from retriver_decider import retriver_decider
from response_generator import generate_response_wt_retrieval, mulitple_response_generation
from chunk_relevance_evaluator import documents_scoring, documents_evaluation, mulitple_response_evaluation
from create_index import EmbeddingManager
from langchain_splitter import split_text


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("Self RAG")

if "index_name" not in st.session_state:
    st.session_state.index_name = None

if "retriver" not in st.session_state:
    st.session_state.retriver = None



# Sidebar : Documents loading and retriever initialization
st.sidebar.header("Documents Loading")

index_name = st.sidebar.text_input("Index Name", key = "index_name")
document = st.sidebar.file_uploader("Upload a document", type = ["pdf"], key = "document")


if st.session_state.index_name is None:
    if st.session_state.retriver is None:
        # split text into chunks
        chunk_text = split_text(document.read().decode("utf-8"), chunk_size = 1000, chunk_overlap = 200)

        # Store chunk into the database
        embedding_manager = EmbeddingManager(index_name = index_name, embedding_size = 1536)
        pc = embedding_manager.get_retriever()
        st.session_state.retriver = pc
        st.session_state.index_name = index_name
        st.success(f"Document uploaded and retriever initialized with index '{index_name}'.")


# Main content area : Query input and submission
query = st.text_input("Enter your query", key = "query_input")
query_button = st.button("Submit", key = "submit_button")


if (st.session_state.index_name is not None) and (st.session_state.retriver is not None):

    if query_button and query :
        # Decide if retrieval is needed
        retriver_decision = retriver_decider(query)

        # When retrieval not needed, generate response directly
        if retriver_decision == "Yes" :
            response = generate_response_wt_retrieval(query)
            st.write("Response:", response)


        else :
            # If retrieval is needed, use the retriever to get relevant documents
            retrieved_docs = st.session_state.retriver.invoke(query)

            # Convert documents to a list of strings
            retrieved_docs = [chunk.page_content for chunk in retrieved_docs]
    
            # Score the relevance of the retrieved documents
            scores = documents_scoring(query = query, documents = retrieved_docs)
    
            # Evaluate the documents based on the scores
            evaluated_docs = documents_evaluation(retrieved_docs, scores, threshold = 0.9)

            if not evaluated_docs :
                response = generate_response_wt_retrieval(query)
                st.write("Response:", response)

            else : 
                # Generate multiple responses based on the evaluated documents
                response_lst = mulitple_response_generation(query = query, documents = evaluated_docs)

                # Evaluate the generated responses for faithfulness
                response_evalution_lst = mulitple_response_evaluation(query = query, responses = response_lst, documents = evaluated_docs)

                # Get index with max faithfulness score
                max_index = np.argmax(response_evalution_lst)

                # Get the final response based on the max faithfulness score
                output = response_lst[max_index]

                st.write("Response:", output)