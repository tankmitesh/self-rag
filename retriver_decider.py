from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI



def retriver_decider(query : str) -> str :

    retrieval_prompt = PromptTemplate(input_variables = ["query"],
                                      template =  """Given the query '{query}', decide if external information is needed.
                                                If the context is enough to answer the query, or if the query is ambiguous or doesn't require retrieval, output 'No'.  
                                                If more information is needed, output 'Yes'.  
                                                Only respond with 'Yes' or 'No'. No explanations."""
                                      )
    
    
    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.0, max_tokens = 10)
    response = llm.invoke(retrieval_prompt.format(query = query))
    response_text = response.content.strip()

    if response_text not in ["Yes", "No"]:
        raise ValueError("Response must be 'Yes' or 'No'.")
    
    return response_text