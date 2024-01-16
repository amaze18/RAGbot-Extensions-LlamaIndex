import os
import openai

openai.api_key=os.environ['SECRET_TOKEN']

from llama_index.extractors.metadata_extractors import EntityExtractor
from llama_index.node_parser import SentenceSplitter

from llama_index import SimpleDirectoryReader
from llama_index.ingestion import IngestionPipeline

from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.memory import ChatMemoryBuffer

#---------------ENTITY EXTRACTOR---------------------#

entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_entities=False,  # include the entity label in the metadata (can be erroneous)
    device="cpu",  # set to "cuda" if you have a GPU
)
#-------------NODE PARSER - DEDUCE TEXTUAL RELATIONSHIPS --------------#

node_parser = SentenceSplitter(chunk_overlap=200,chunk_size=2000)

transformations = [node_parser, entity_extractor]

#-------------------------LOAD DATA ------------------------------------#
documents = SimpleDirectoryReader(input_dir=r"scraped_files\processed\striped_files").load_data()
pipeline = IngestionPipeline(transformations=transformations)

#-----------------------GENERATE NODES, SERVICE CONTEXT AND NODES --------------------------#
nodes = pipeline.run(documents=documents)

service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2))

index = VectorStoreIndex(nodes, service_context=service_context)
index.storage_context.persist("BITSPilani/")


#-----------------------------CHATBOT FUNCTIONS------------------------------#
##-----------------------------Add a proper prompt here ----------------------#
### ------Functions adopted from https://github.com/RahulSundar/ChatBotProject/blob/main/chatbotfunctions.py -------#


def react_chatbot_engine(index):

    #memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    chat_engine = index.as_chat_engine(
    chat_mode="react",
    #memory=memory,
    system_prompt=(
        "You are a helpful and friendly chatbot who addresses <your requirement here>"
        ),
    verbose=True,
    )
    return chat_engine

def condense_question_chatbot_engine(index):

    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    memory=memory,
    system_prompt=(
        "You are a helpful and friendly chatbot who addresses <your requirement here>"
        ),
    verbose=True,
    )
    return chat_engine

def condense_context_question_chatbot_engine(index):

    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    system_prompt=(
        "You are a helpful and friendly chatbot who addresses <your requirement here>"
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
        "You are a helpful and friendly chatbot who addresses <your requirement here>"
        ),
    )
    return chat_engine
