
"""ISB chatbot.ipynb
Original file is located at
   https://colab.research.google.com/drive/1GYmsZSR4MWuvORNpSWFWrXz79lQKb6oc
"""

"""# Scrape"""

# Regex to match a URL
# HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Define root domain to crawl
domain = "i-venture.org"
sitemap_url = "https://i-venture.org/sitemap.xml"
full_url = "https://i-venture.org/"
import os
RESULTS_DIR = "scraped_files/"
os.makedirs(RESULTS_DIR, exist_ok=True)
import gradio as gr
import random
import time
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import numpy as np
import langchain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline
import glob
from langchain.vectorstores import FAISS
import os
import openai
from dotenv import load_dotenv
'''
#scraping data starts
def get_sitemap(url=sitemap_url):
   try:
       with urllib.request.urlopen(url) as response:
           xml = BeautifulSoup(response,
                               'lxml-xml',
                               from_encoding=response.info().get_param('charset'))

       urls = xml.find_all("url")
       locs = []

       for url in urls:

           if xml.find("loc"):
               loc = url.findNext("loc").text
               locs.append(loc)

       return locs
   except Exception as e:
       print(e)
       return []


def crawl(url):
   # Parse the URL and get the domain
   # local_domain = urlparse(url).netloc

   queue = deque(get_sitemap())

   os.makedirs(RESULTS_DIR + "text/", exist_ok=True)
   os.makedirs(RESULTS_DIR + "processed", exist_ok=True)

   # While the queue is not empty, continue crawling
   while queue:
       # Get the next URL from the queue
       url = queue.pop()
       print(url) # for debugging and to see the progress

       # Save text from the url to a <url>.txt file
       with open(f'{RESULTS_DIR}text/'+ url.strip("/").replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:

           soup = BeautifulSoup(requests.get(url).text, "html.parser")
           text = soup.get_text()

           # If the crawler gets to a page that requires JavaScript, it will stop the crawl
           if ("You need to enable JavaScript to run this app." in text):
               print("Unable to parse page " + url + " due to JavaScript being required")

           f.write(text)

       # # Get the hyperlinks from the URL and add them to the queue
       # for link in get_domain_hyperlinks(local_domain, url):
       #     if link not in seen:
       #         queue.append(link)
       #         seen.add(link)

def remove_newlines(serie):
   serie = serie.str.replace('\n', ' ')
   serie = serie.str.replace('\\n', ' ')
   serie = serie.str.replace('  ', ' ')
   serie = serie.str.replace('  ', ' ')
   return serie


def get_df():
   # Create a list to store the text files
   texts=[]

   for file in os.listdir(RESULTS_DIR + "text/"):
       with open(RESULTS_DIR + "text/" + "/" + file, "r", encoding="UTF-8") as f:
           text = f.read()

           # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
           texts.append((file.replace('#update',''), text))

   # Create a dataframe from the list of texts
   df = pd.DataFrame(texts, columns = ['fname', 'text'])

   # Set the text column to be the raw text with the newlines removed
   df['text'] = df.fname + ". " + remove_newlines(df.text)
   return df
#scraping data ends

## Clean data for openai
'''
#For openai embedding
import tiktoken
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('scraped_files/processed/scraped_new.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()

max_tokens = 500

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

   # Split the text into sentences
   sentences = text.split('. ')

   # Get the number of tokens for each sentence
   n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

   chunks = []
   tokens_so_far = 0
   chunk = []

   # Loop through the sentences and tokens joined together in a tuple
   for sentence, token in zip(sentences, n_tokens):

       # If the number of tokens so far plus the number of tokens in the current sentence is greater
       # than the max number of tokens, then add the chunk to the list of chunks and reset
       # the chunk and tokens so far
       if tokens_so_far + token > max_tokens:
           chunks.append(". ".join(chunk) + ".")
           chunk = []
           tokens_so_far = 0

       # If the number of tokens in the current sentence is greater than the max number of
       # tokens, go to the next sentence
       if token > max_tokens:
           continue

       # Otherwise, add the sentence to the chunk and add the number of tokens to the total
       chunk.append(sentence)
       tokens_so_far += token + 1

   # Add the last chunk to the list of chunks
   if chunk:
       chunks.append(". ".join(chunk) + ".")

   return chunks

def shorten(df):
   shortened = []

   # Loop through the dataframe
   for row in df.iterrows():

       # If the text is None, go to the next row
       if row[1]['text'] is None:
           continue

       # If the number of tokens is greater than the max number of tokens, split the text into chunks
       if row[1]['n_tokens'] > max_tokens:
           shortened += split_into_many(row[1]['text'])

       # Otherwise, add the text to the list of shortened texts
       else:
           shortened.append( row[1]['text'] )

   new_df = pd.DataFrame(shortened, columns = ['text'])
   new_df['n_tokens'] = new_df.text.apply(lambda x: len(tokenizer.encode(x)))
   return new_df

df = shorten(df)
df.n_tokens.hist()

"""# QnA"""

from ast import literal_eval
load_dotenv()
SECRET_IN_ENV = True
SECRET_TOKEN = os.environ['SECRET_TOKEN']
openai.api_key = SECRET_TOKEN
def create_context(
   question, df=df
):   # pass opensource as argument
   """
   Create a context for a question by finding the most similar context from the dataframe
   """
   max_len=5000
   size="ada"
   # Get the embeddings for the question
   # print("question::",question)
   #st.write(question)
   q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding'] #use open_source embedding
   df = pd.read_csv('embeddings.csv', index_col=0)
   df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
   # Get the distances from the embeddings
   df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


   returns = []
   cur_len = 0

   # Sort by distance and add the text to the context until the context is too long
   for i, row in df.sort_values('distances', ascending=True).iterrows():

       # Add the length of the text to the current length
       cur_len += row['n_tokens'] + 4

       # If the context is too long, break
       if cur_len > max_len:
           break

       # Else add it to the text that is being returned
       returns.append(row["text"])

   # Return the context
   return "\n\n###\n\n".join(returns)

def answer_question(

   question,df=df

):


   """
   Answer a question based on the most similar context from the dataframe texts
   """
   open_source=0
   model="text-davinci-003"
   max_len=1800
   size="ada"
   debug=False
   max_tokens=250
   stop_sequence=None
   try:
      open_source=1
      # code for opensource LLM
      if open_source==0:


          model_name = "sentence-transformers/all-mpnet-base-v2"
          model_name = "sentence-transformers/LaBSE"
          model_name= 'intfloat/e5-large-v2'
          model_name = 'all-MiniLM-L6-v2'
          model_name='google/flan-ul2'
          model_name='google/flan-t5-small'
          model_kwargs = {'device': 'cpu'}
          encode_kwargs = {'normalize_embeddings': False}
          hf = HuggingFaceEmbeddings(
                     model_name="sentence-transformers/all-mpnet-base-v2",
                     model_kwargs=model_kwargs,
                     encode_kwargs=encode_kwargs
                 )
          #docs is documents that need to be converted to word embeddings
          documents = []
          a=glob.glob("scraped_files/processed/scraped_new.csv")
          for i in range(len(a)):
            documents.extend(CSVLoader(a[i]).load())
          chunk_size = 1024
          text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
          texts = text_splitter.split_documents(documents)

          db = Chroma.from_documents(texts, hf)

          retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 3} )
          chunk_size = 2048
          model_id="sentence-transformers/all-MiniLM-L6-v2"
          model_id='digitous/Alpacino30b'
          model_id="google/flan-t5-base"
          model_id='tiiuae/falcon-40b'
          model_id="google/flan-t5-small" #base

          llm =  langchain.llms.huggingface_pipeline.HuggingFacePipeline.from_model_id(model_id="google/flan-t5-small", task="text2text-generation", model_kwargs={"max_length" : chunk_size})
          qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
          res = qa(question)
          answer, docs = res['result'], res['source_documents']
          # Get context from all the relevant sources used
          doc_list=[]
          for document in docs:

                  metadata=document.metadata["source"]
                  doc_list.append(document)
          return answer

      else:
         

         #generate openai emebeddings
         context = create_context(
             question,
             df,)
         # If debug, print the raw model response
         if debug:
             print("Context:\n" + context)
             print("\n\n")
         introduction = 'Use the below text to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
         question_ai = f"\n\nQuestion: {question}"
         message = introduction
         message = message + context + question_ai
         messages = [
             {"role": "system","content": "You are iVBot, an AI based chatbot assistant. You are friendly, proactive, factual and helpful, \
             you answer from the context provided"}, {"role": "user", "content": message},
         ]
         response = openai.ChatCompletion.create(
         model='gpt-3.5-turbo',
             messages=messages,
             temperature=0.01,
             top_p=0.75)
         answer = response["choices"][0]["message"]["content"]
          # Create a completions using the questin and context
         return answer #response["choices"][0]["text"].strip()
   except Exception as e:
       print(e)
       return e
