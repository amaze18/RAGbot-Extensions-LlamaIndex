import openai
import getpass
import os
import faiss
import warnings
warnings.filterwarnings("ignore")
openai.api_key = os.environ['SECRET_TOKEN']
import langchain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import re
text_folder_path = r"scraped_files/processed"
texts=[]

for file in os.listdir(text_folder_path):
    try:
        with open(text_folder_path+ "/" + file, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                if re.search('\S', line): 
                    print(line)
        f.close()
    except UnicodeDecodeError:
        with open(text_folder_path+ "/" + file, "r", encoding="latin-1") as f:
            for line in f.readlines():
                if re.search('\S', line): 
                    print(line)
        f.close()
    else:
        f.close()

text_loader_kwargs={'autodetect_encoding': True}
loader=DirectoryLoader(text_folder_path,glob="./*.txt",use_multithreading=True,loader_kwargs=text_loader_kwargs,silent_errors=True)
documents=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=120)
texts=text_splitter.split_documents(documents)
print(texts)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=openai.api_key)
db=FAISS.from_documents(texts,embeddings)


def chat_gpt(question):

    embeddings= OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)



    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 3} )#do not increase k beyond 3, else
    llm = OpenAI(model='text-embedding-ada-002',temperature=0, openai_api_key=openai.api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


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
    return answer
