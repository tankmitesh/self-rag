from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRelevance, Faithfulness
from ragas.dataset_schema import SingleTurnSample


async def documents_scoring(query: str, documents: list) -> list:

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    context_relevance_method = ContextRelevance(llm=evaluator_llm)

    score_lst = []
    for xdocument in documents:
        sample = SingleTurnSample(user_input = query, retrieved_contexts = [xdocument])
        score = await context_relevance_method.single_turn_ascore(sample)
        score_lst.append(score)
        
    return score_lst


def documents_evaluation(documents : list, scores: list, threshold = 0.9) -> list :

    evaluated_docs = []
    for doc, score in zip(documents, scores):
        if score >= threshold:
            evaluated_docs.append(doc)
    
    return evaluated_docs


def mulitple_response_evaluation(query : str, responses : list, documents : str) :

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    faithfulness_method = Faithfulness(llm=evaluator_llm)

    score_lst = [ ]
    for xresponse, xdocument in zip(responses, documents):
        faithfulness_score = faithfulness_method.score({'user_input': query, 
                                                        'retrieved_contexts': [xdocument], 
                                                        'response': xresponse})
        score_lst.append(faithfulness_score)
        
    return score_lst    