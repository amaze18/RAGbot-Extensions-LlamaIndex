import os
import openai
openai.api_key=os.environ['SECRET_TOKEN']
from llama_index.legacy import SimpleDirectoryReader
from llama_index.legacy.extractors.metadata_extractors import EntityExtractor
from llama_index.legacy.node_parser import SentenceSplitter
from llama_index.legacy.ingestion import IngestionPipeline
from llama_index.legacy import ServiceContext, VectorStoreIndex
from llama_index.legacy.llms import OpenAI
from llama_index.legacy import (StorageContext,load_index_from_storage)
from llama_index.legacy.embeddings import OpenAIEmbedding
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
def indexgenerator(indexPath, documentsPath):
    
    # check if storage already exists
    if not os.path.exists(indexPath):
        print("Not existing")
        # load the documents and create the index
        
        entity_extractor = EntityExtractor(prediction_threshold=0.2,label_entities=False, device="cpu")

        node_parser = SentenceSplitter(chunk_overlap=102,chunk_size=1024)

        transformations = [node_parser, entity_extractor]

        documents = SimpleDirectoryReader(input_files=[documentsPath]).load_data()

        pipeline = IngestionPipeline(transformations=transformations)

        nodes = pipeline.run(documents=documents)

        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0),embed_model=embed_model)

        index = VectorStoreIndex(nodes, service_context=service_context)

        # store it for later
        index.storage_context.persist(indexPath)
    else:
        print("Existing")
        storage_context = StorageContext.from_defaults(persist_dir=indexPath)
        index = load_index_from_storage(storage_context,service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0),embed_model=embed_model))
        
    return index

