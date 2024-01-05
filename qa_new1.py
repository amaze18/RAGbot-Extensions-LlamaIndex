import openai
import getpass
import os
SECRET_TOKEN = os.environ["SECRET_TOKEN"] 
openai.api_key = SECRET_TOKEN

#openai.api_key="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"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
text_folder_path = r"C:\Users\Kush Juvekar\Downloads\scraped_files_new\scraped_files\text"
print(os.listdir(text_folder_path))

# Load multiple files

# Specify the desired encodings
desired_encodings = ['utf-8', 'latin-1']

# Create an empty list to store TextLoader instances
loaders = []

# Loop through each file in the text folder
for fn in os.listdir(text_folder_path):
    file_path = os.path.join(text_folder_path, fn)
    
    # Try loading with each encoding
    for encoding in desired_encodings:
        try:
            # Attempt to create a TextLoader instance with the current encoding
            loader = TextLoader(file_path, encoding=encoding)
            loaders.append(loader)
            break  # Break out of the inner loop if successful
        except (UnicodeDecodeError):
            # Handle the exception (file couldn't be decoded with the current encoding)
            print(f"Failed to load {file_path} with encoding {encoding}")
all_documents = []
# List of suffixes to remove
def get_answer(question):
    for loader in loaders:
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(raw_documents)
        all_documents.extend(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(all_documents, embeddings)
    answer=db.similarity_search(question)
    return answer
