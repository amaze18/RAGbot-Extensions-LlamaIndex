import openai
import os
import faiss
import warnings
warnings.filterwarnings("ignore")
openai.api_key = os.environ['SECRET_TOKEN']
import langchain
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import DirectoryLoader
#from langchain_community.document_loaders import TextLoader
#from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
#from sqlalchemy.exc import InvalidRequestError
from openai.error import InvalidRequestError
from langchain_community.vectorstores import FAISS
from langchain import OpenAI
#from langchain.chat_models import ChatOpenAI
#from langchain.chains import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
#from langchain.prompts import PromptTemplate
#from langchain.prompts import SystemMessagePromptTemplate
#from langchain.prompts import HumanMessagePromptTemplate
#from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import re
"""
text_folder_path = r"scraped_files/processed"
texts=[]

for file in os.listdir(text_folder_path):
    try:
        with open(text_folder_path+ "/" + file, "r", encoding="UTF-8") as f,open(text_folder_path+"/"+"striped_files/"+file,"w",encoding="UTF-8") as o:
            for line in f:
                if line.strip():
                    o.write(line)
    except UnicodeDecodeError:
        with open(text_folder_path+ "/" + file, "r", encoding="latin-1") as f,open(text_folder_path+"/"+"striped_files/"+file,"w",encoding="latin-1") as o:
            for line in f:
                if line.strip():
                    o.write(line)
  
text_loader_kwargs={'autodetect_encoding': True}
loader=DirectoryLoader(text_folder_path+"/"+"striped_files",glob="./*.txt",use_multithreading=True,loader_cls=TextLoader,silent_errors=True,loader_kwargs=text_loader_kwargs)
documents=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
texts=text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=openai.api_key)
db=Chroma.from_documents(texts,embeddings,persist_directory="scraped_files/processed")
db=None
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
"""
def chat_gpt(question):
    embeddings= OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=openai.api_key)
    db=FAISS.load_local("scraped_files/processed",embeddings)
    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 3} )
    llm = OpenAI(openai_api_key=openai.api_key)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever, return_source_documents=True)
    query = question
    res = qa(query)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    #model="gpt-4-0613",
    messages=[
        {"role": "system", "content": "You are a helpful chatbot who answers questions asked based only on context provided in a friendly tone, if you do not know the answer, say I don\'t know"},
        {"role": "user", "content": f"{res}"}
        ])
    answer= response["choices"][0]["message"]["content"]
    return [answer,res]