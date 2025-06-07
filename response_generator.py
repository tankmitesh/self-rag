import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def generate_response_wt_retrieval(query: str) -> str :

    response_prompt = PromptTemplate(input_variables=["query"],
                                     template="""Consider yourself as a helpful assistant.
                                                Give a detailed answer to the query '{query}'."""
                                     )
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    response = llm.invoke(response_prompt.format(query=query))
    
    return response.content.strip()


def mulitple_response_generation(query : str, documents : list) -> list :

    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.0)

    response_list = [ ]
    for doc in (documents):
        response_prompt = PromptTemplate(input_variables = ["query", "document"],
                                    template = """consider yourself as a helpful assistant.
                                                Given the query '{query}', and based on the document '{document}', generate a detailed response.
                                                Provide a comprehensive answer that addresses the query using the information from the document.
                                                ONLY respond with the generated response.
                                                """
                                    )
        
        response = llm.invoke(response_prompt.format(query = query, document = doc)).content.strip()
        response_list.append(response)

    return response_list