import os
import openai
openai.api_key=os.environ['SECRET_TOKEN']

from llama_index.extractors.metadata_extractors import EntityExtractor
from llama_index.node_parser import SentenceSplitter

entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_entities=False,  # include the entity label in the metadata (can be erroneous)
    device="cpu",  # set to "cuda" if you have a GPU
)

node_parser = SentenceSplitter(chunk_overlap=200,chunk_size=2000)

transformations = [node_parser, entity_extractor]

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_dir=r"scraped_files\processed\striped_files").load_data()
from llama_index.ingestion import IngestionPipeline

pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(documents=documents)
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI

service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2))

index = VectorStoreIndex(nodes, service_context=service_context)
index.storage_context.persist("BITSPilani/")