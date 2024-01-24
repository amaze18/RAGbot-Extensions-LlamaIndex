#-------------STD DEPENDENCIES---------------------------#
import os
from collections import namedtuple
import math

#---------------CHATBOT DEPENDENCIES-----------------#
import altair as alt
import pandas as pd
import openai
#from hugchat import hugchat
#from hugchat.login import Login
#import openpyxl
from llama_index import StorageContext, load_index_from_storage
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.postprocessor import MetadataReplacementPostProcessor
from qa_llamaindex import indexgenerator
from create_context import generate_response
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import LongContextReorder
from llama_index.schema import Node, NodeWithScore
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.retrievers import BaseRetriever
from llama_index.retrievers import BM25Retriever
from qa_llamaindex import react_chatbot_engine, condense_context_question_chatbot_engine, context_chatbot_engine,condense_question_chatbot_engine
from qa_llamaindex import react_chatbot_engine, condense_question_chatbot_engine, condense_context_question_chatbot_engine, context_chatbot_engine
#----------------------UI DEPENDENCIES---------------#
from PIL import Image

import streamlit as st

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: display;}
      footer {visibility: display;}
     .stApp { bottom: 105px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 50, 0, 50),
        width=percent(100),
        color="black",
        text_align="left",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(1.5)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)



SECRET_TOKEN = os.environ["SECRET_TOKEN"]
openai.api_key = SECRET_TOKEN

# App title
st.set_page_config(page_title="🤗💬 I-Venture @ ISB AI-Chat Bot")
st.header("I-Venture @ ISB AI-Chat Bot")

# Hugging Face Credentials
with st.sidebar:
    st.title('🤗💬I-Venture @ ISB Chat Bot')
    st.success('Access to this Gen-AI Powered Chatbot is provided by  [Anupam](https://www.linkedin.com/in/anupamisb/)!!', icon='✅')
    hf_email = 'anupam_purwar2019@pgp.isb.edu'
    hf_pass = 'PASS'
    st.markdown('📖 This app is hosted by I-Venture @ ISB [website](https://i-venture.org/)!')
    image = Image.open('Ivlogo.png.png')
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

#storage_context = StorageContext.from_defaults(persist_dir=indexPath)
#index = load_index_from_storage(storage_context)
indexPath="llamaindex_entities_0.2"
documentsPath="scraped_files\processed\striped_files_new"
index=indexgenerator(indexPath,documentsPath)
vector_retriever = index.as_retriever(similarity_top_k=10)
nodes=index.docstore.docs.values()
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes
hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
# User-provided prompt
page_bg_img = '''
<style>
body {
background-image: url("https://csrbox.org/media/Hero-Image.png");
background-size: cover;
}
</style>
'''
if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Ask anything about I-Venture @ ISB ..."}]

    # Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
from llama_index.postprocessor import LongContextReorder
from llama_index.schema import Node, NodeWithScore
from llama_index.response_synthesizers import get_response_synthesizer

response_synthesizer = get_response_synthesizer(response_mode="refine")
postprocessor = LongContextReorder()
qa_prompt="You are a helpful and friendly chatbot who addresses queries regarding I-Venture @ ISB.Instruction: Use the previous chat history, or the context above, to interact and help the user.If you can find the answer you say I don't know"
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
                    #base_retriever = index.as_retriever(similarity_top_k=5)
                    llm = OpenAI(model="gpt-3.5-turbo")
                    service_context = ServiceContext.from_defaults(llm=llm)
                    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=10)
                    query_engine_base = RetrieverQueryEngine.from_args(hybrid_retriever, service_context=service_context,node_postprocessors=[reranker,MetadataReplacementPostProcessor(target_metadata_key="window"),postprocessor],response_synthesizer=response_synthesizer,qa_prompt=qa_prompt)
                    response = query_engine_base.query(prompt)
                    if "not mentioned in" in response.response or "sorry" in response.response or "I don't know" in response.response:
                        st.write("DISTANCE APPROACH")
                        response=generate_response(prompt,hf_email,hf_pass)
                        st.write(response)
                        message = {"role": "assistant", "content": response}
                        st.session_state.messages.append(message)
                    else:
                        st.write(response.response)
                        message = {"role": "assistant", "content": response.response}
                        st.session_state.messages.append(message)

myargs = [
    "Made in India",""
    " with ❤️ by ",
    link("https://www.linkedin.com/in/anupamisb/", "@Anupam"),
     br(),
     link("https://i-venture.org/chatbot/", "ISB ChatBoT"),
    ]

def footer():
    myargs = [
    "Made in India",""
    " with ❤️ by ",
    link("https://www.linkedin.com/in/anupamisb/", " Anupam for "),
    link("https://i-venture.org/chatbot/", "I-Venture @ ISB"),
    ]
    layout(*myargs)

#layout(*myargs)
footer()
