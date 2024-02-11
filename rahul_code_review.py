from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from ipywidgets.widgets import FloatSlider
from IPython.display import display
from llama_index.memory import ChatMemoryBuffer
import openai
from create_context import answer_question
import time
from PIL import Image
from llama_index.retrievers import BM25Retriever
from llama_index.retrievers import VectorIndexRetriever
import boto3
from io import StringIO
#from hugchat import hugchat
#from hugchat.login import Login
from llama_index.schema import MetadataMode
from rouge import Rouge
import os
from llama_index.postprocessor import SentenceTransformerRerank
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
import tiktoken
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
indexPath=r"index\1024\text_embedding_ada_002"
documentsPath=r"Text_Files_Old"
index=indexgenerator(indexPath,documentsPath)
vector_retriever = VectorIndexRetriever(index=index,similarity_top_k=2)
#nodes=index.docstore.docs.values()
bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=2)
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
context_prompt=(
        "You are a helpful and friendly chatbot who addresses queries regarding I-Venture @ ISB."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction 1: Use the previous chat history, or the context above to answer. Be concise."
        "\nInstruction 2: Say I don't know if you do not find the answer in context provided."
        )
memory=ChatMemoryBuffer.from_defaults(token_limit=3900)
rouge = Rouge()


def get_response(prompt):
    llm = OpenAI(model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)
    query_engine=RetrieverQueryEngine.from_args(retriever=hybrid_retriever,service_context=service_context)
    chat_engine=CondensePlusContextChatEngine.from_defaults(query_engine,memory=memory,context_prompt=context_prompt)
    nodes = hybrid_retriever.retrieve(str(prompt))
    context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes])
    response = chat_engine.chat(str(prompt))
    validating_prompt = ("""You are an intelligent bot designed to assist users on an organization's website by answering their queries. You'll be given a user's question and an associated answer. Your task is to determine if the provided answer effectively resolves the query. If the answer is unsatisfactory, return 0.\n
                        Query: {question}  
                        Answer: {answer}
                        Your Feedback:
                        """)
    feedback = llm.complete(validating_prompt.format(question=prompt,answer=response.response))
    if feedback.text==str(0):
        st.write("DISTANCE APPROACH")
        response , joined_text=answer_question(prompt)
        st.write(response)
        scores = rouge.get_scores(response, joined_text)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)                                            
    else:
        scores=rouge.get_scores(response.response,context_str)
        message = {"role": "assistant", "content": response.response}
        st.session_state.messages.append(message)
        response_list = [response.response , prompt , scores]
        return response_list
if "query_status" not in st.session_state:
    st.session_state["query_status"] = False
if "messages" not in st.session_state.keys():
    st.session_state.messages = []
with st.chat_message("user"):
    prompt = st.chat_input("Let me know what you have in mind!")
st.session_state.messages.append({"role": "user", "content": prompt})
if prompt:
    st.session_state["query_status"] = True
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_list = get_response(prompt=prompt)
            st.write(response_list[0])
        with st.form("kjbjb",clear_on_submit=True):
            answer_quality = st.slider("Rate the answer out of 5: ",0.0,5.0,step=0.5)
            if submitted :=st.form_submit_button("Submit"):
                df = pd.read_csv('logs/conversation_logs.csv')
                new_row = {'question': response_list[1], 'answer': response_list[0], 'Answer Quality' : answer_quality, 'Expected Answer': '','Rouge_Score' : response_list[2]}
                df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
                df.to_csv('logs/conversation_logs.csv', index=False)
                bucket = 'aiex' # already created on S3
                csv_buffer = StringIO()
                df.to_csv(csv_buffer)
