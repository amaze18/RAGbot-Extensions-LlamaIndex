import os
import openai
openai.api_key=os.environ['SECRET_TOKEN']
from llama_index import SimpleDirectoryReader
from llama_index.extractors.metadata_extractors import EntityExtractor
from llama_index.node_parser import SentenceSplitter
from llama_index.ingestion import IngestionPipeline
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI
import streamlit as st 
from llama_index import (StorageContext,load_index_from_storage)
from llama_index.memory import ChatMemoryBuffer
@st.cache_resource
def indexgenerator(indexPath, documentsPath):

    # check if storage already exists
    if not os.path.exists(indexPath):
        print("Not existing")
        # load the documents and create the index
        
        entity_extractor = EntityExtractor(prediction_threshold=0.2,label_entities=False, device="cpu")

        node_parser = SentenceSplitter(chunk_overlap=200,chunk_size=2000)

        transformations = [node_parser, entity_extractor]

        documents = SimpleDirectoryReader(input_dir=r"scraped_files\processed\striped_files").load_data()

        pipeline = IngestionPipeline(transformations=transformations)

        nodes = pipeline.run(documents=documents)

        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0))

        index = VectorStoreIndex(nodes, service_context=service_context)

        # store it for later
        index.storage_context.persist(indexPath)
    else:
        #load existing index
        print("Existing")
        storage_context = StorageContext.from_defaults(persist_dir=indexPath)
        index = load_index_from_storage(storage_context)
        
    return index

def react_chatbot_engine(index):

    #memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    chat_engine = index.as_chat_engine(
    chat_mode="react",
    #memory=memory,
    system_prompt=(
        "You are a helpful and friendly chatbot who addresses questions regarding I-Venture @ ISB politely."
        ),
    verbose=True,
    )
    return chat_engine

def condense_question_chatbot_engine(index):

    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    memory=memory,
    #system_prompt=(
       # "You are a helpful and friendly chatbot who addresses queries regarding I-Venture @ ISB."
        #),
    verbose=True,
    )
    return chat_engine

def condense_context_question_chatbot_engine(index):

    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    system_prompt=(
        "You are a helpful and friendly chatbot who addresses queries regarding I-Venture @ ISB."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
        ),
    verbose=True,
    )
    return chat_engine

def context_chatbot_engine(index):

    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a helpful and friendly chatbot who addresses queries regarding I-Venture @ ISB."
        ),
    )
    return chat_engine
