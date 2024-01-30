from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from datetime import timedelta
from llama_index.memory import ChatMemoryBuffer
import openai
from create_context import answer_question
from PIL import Image
from llama_index.retrievers import BM25Retriever
#from hugchat import hugchat
#from hugchat.login import Login
import os
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
from qa_llamaindex import indexgenerator
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index.query_engine import RetrieverQueryEngine

from llama_index import (get_response_synthesizer)

from llama_index.query_engine import RetrieverQueryEngine

from llama_index.chat_engine import CondensePlusContextChatEngine
# import QueryBundle

# Retrievers
from llama_index.retrievers import (BaseRetriever)

from typing import List
#from llama_index import (VectorStoreIndex,SimpleDirectoryReader)
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
indexPath="scraped_files\processed\striped_files_new\llamaindex_entities_0.2"
documentsPath="scraped_files\processed\striped_files_new"
index=indexgenerator(indexPath,documentsPath)
vector_retriever = index.as_retriever(similarity_top_k=2)
nodes=index.docstore.docs.values()
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
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
            if n.node.node_id not in node_ids: #apply processing at this very stage
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
        st.markdown(message["content"])
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
                    #nodes=index.docstore.docs.values()
                    llm = OpenAI(model="gpt-3.5-turbo")
                    service_context = ServiceContext.from_defaults(llm=llm)
                    query_engine=RetrieverQueryEngine.from_args(retriever=hybrid_retriever,service_context=service_context)
                    nodes=query_engine.retrieve(str(prompt))
                    context_list=[]
                    for n in nodes:
                         context_list.append(n.get_content())
                    context_str=""
                    for i in range(len(context_list)):
                         context_str+="Document "+str(i)+context_list[i]
                    chat_engine=CondensePlusContextChatEngine.from_defaults(query_engine,memory=ChatMemoryBuffer(token_limit=3900),system_prompt=(
        "You are a helpful and friendly chatbot who addresses queries regarding I-Venture @ ISB."
        "\nInstruction: Store context for only last 2 questions."
        ),context_prompt="""
  The following is a friendly conversation between a user and an AI assistant.
  The assistant is talkative and provides lots of specific details from its context.
  If the assistant does not know the answer to a question, it truthfully says it
  does not know.

  Here are the relevant documents for the context:

  {context_str}

  Instruction: Store context for only last 2 questions.
  """)
                    response = chat_engine.chat(str(prompt))
                    if "not mentioned in" in response.response or "I don't know" in response.response:
                        st.write("DISTANCE APPROACH")
                        response=answer_question(prompt)
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
