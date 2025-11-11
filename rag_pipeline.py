from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
import dotenv
import os
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker, RerankRequest
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Optional
from langgraph.graph import StateGraph
import pandas as pd

class RAGState(TypedDict):
    query: Optional[str]
    docs: Optional[list]
    answer: Optional[str]
    pages: Optional[list]

class RAGAgent:
    def __init__(self, retrieve_k = 10, context_k = 5):
        dotenv.load_dotenv()

        embeddings = OllamaEmbeddings(
            model="jeffh/intfloat-multilingual-e5-small:f32",
        )

        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-oss-20b:free"
        )

        vectorstore = Chroma(
            collection_name="websites_data",
            embedding_function=embeddings,
            persist_directory="./alpha_chroma_db"
        )

        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n = context_k)
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": retrieve_k})
        self.retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
        #self.context_k = context_k

        self.systemprompt = '''Ты профессиональный банковский помощник. Твоя задача давать клиенту правидивую и релевантную информацию по поводу банковских услуг.
                Тебе будут даны вопрос пользователя и контекст по 5 документам, в которых дана информация по теме вопроса
                Старайся брать ответы только из этих документов. Отвечай только на русском и соблюдай профессиональную этику'''
        self.context = '''контекст: 1. {}
                            2. {}
                            3. {}
                            4. {}
                            5. {}'''
        self.query = 'вопрос пользователя: {}'

        self.prompt = ChatPromptTemplate(
            [MessagesPlaceholder("systemprompt"),
            MessagesPlaceholder("context"),
            MessagesPlaceholder("query")]
        )

        builder = StateGraph(RAGState)
        builder.add_node("input", self.input_node)
        builder.add_node("retriever", self.get_relevant_docs)
        builder.add_node("output", self.get_answer)

        builder.set_entry_point("input")
        builder.add_edge("input", "retriever")
        builder.add_edge("retriever", "output")
        builder.set_finish_point("output")
        self.graph = builder.compile()
    
    def input_node(self, state: RAGState):
        print(state['query'])
        return state
    
    def get_relevant_docs(self, state: RAGState):
        query = state['query']
        docs = self.retriever.invoke(f'query: {query}')
        pages = [doc.metadata['web_id'] for doc in docs]
        return {'docs': docs, 'pages': pages}

    def get_answer(self, state: RAGState):
        current_context = [doc.page_content for doc in state['docs']]
        msg = self.llm.invoke(self.prompt.invoke(
            {"systemprompt": [SystemMessage(self.systemprompt)],
            "context": [SystemMessage(self.context.format(*current_context))],
            "query": [HumanMessage(self.query.format(state['query']))]}
        ))
        answer = msg.content
        print(answer)
        return {'answer': answer}
    
    def ask_question(self, query):
        output = self.graph.invoke({'query': query})
        return {'answer': output['answer'], 'pages': output['pages']}

if __name__ == '__main__':
    agent = RAGAgent()
    
    df = pd.read_csv('./data/questions_clean.csv')
    pages = []
    for i, row in df.iterrows():
        output = agent.ask_question(row.query)
        pages.append(str(output['pages']))
    
    subm = pd.read_csv('./data/sample_submission.csv')
    subm.web_list = pages
    subm.to_csv('./subm/subm.csv', index = False)


