import os
import textwrap
from dotenv import find_dotenv, load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import get_all_tool_names

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings()

### Input: The Youtube Video URL
### Output: A vector Store based on the transcripts of the input youtube video

def create_db_from_youtube_video_url(video_url):
    ##Loading Youtube Transcripts as documents
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    ##Splitting the documents into chunks, with a chunk_size and overlap strategy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap =100)
    docs = text_splitter.split_documents(transcript)
    
    ## Creating a Vector store client based on th documents and chuks
    db = FAISS.from_documents(docs, embeddings)
    
    return db

### Input: The vector store created in the last step, Query(Input) and k (number of closest embeddings to Query)
### Output: Response to the Query, k closest docs joined together(for model transparency and Understanding)

def get_response_from_query(db, query, k=4):
    
    ## Finding the k closest vector embeddings to the query in the VS
    docs = db.similarity_search(query, k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    
    ## Initializing a ChatModel 
    chat = ChatOpenAI(
        model = "gpt-3.5-turbo"
        ,api_key="sk-None-NmK2oXjPHTSresy9n630T3BlbkFJHXNgwZkFNRNVfUIX75lN"
        ,temperature=0.2
    )
    chunks = []
    
    ##Initializing a template
    template = """
                You are a helpful assistant that can answer questions about the youtube videos based on the video's transcript: {docs}
                
                Only use the factual information from the transcript to answer the question.
                
                If you feel like you don't have enough information to answer the question, say 'I don't know'
                
                Your answers should be verbosed and detailed.
                """
    ## Initializing system prompt
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    
    
    ## Initializing Human prompt
    human_template = "Answer the following question {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    ## Creating a chatpromot based on Human and System prompt
    chat_prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt])
    
    ## Chaining everything
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    
    ## Getting the response to the query
    response = chain.run(question= query, docs= docs_page_content)
    response = response.replace("\n","")
       
    
    return response, docs
    
## Sam Altman: OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power & AGI | Lex Fridman Podcast 
video_url = "https://www.youtube.com/watch?v=jvqFAi7vkBc" 

db = create_db_from_youtube_video_url(video_url)

query = "Give me a short summary of the video"



    
response, docs = get_response_from_query(db, query)

print(textwrap.fill(response, width=85))