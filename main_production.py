import warnings
warnings.filterwarnings('ignore')

import streamlit as st

import os

import openai

import pandas as pd
from PIL import Image
import boto3
from io import StringIO
from rouge import Rouge
import tiktoken

from htbuilder import HtmlElement, div, br, hr, a, p, img, styles
from htbuilder.units import percent, px

from qa_llamaindex import indexgenerator
from create_context import answer_question

from llama_index.legacy.memory import ChatMemoryBuffer
from llama_index.legacy.retrievers import BM25Retriever
from llama_index.legacy.retrievers import VectorIndexRetriever
from llama_index.legacy.core.llms.types import ChatMessage, MessageRole
from llama_index.legacy.schema import QueryBundle
from llama_index.legacy.schema import MetadataMode
from llama_index.legacy.postprocessor import LongContextReorder 
from llama_index.legacy.llms import OpenAI
from llama_index.legacy import ServiceContext
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.chat_engine import CondensePlusContextChatEngine
from llama_index.legacy.retrievers import BaseRetriever

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



openai.api_key =os.environ['SECRET_TOKEN']

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

#nodes=index.docstore.docs.values()

indexPath_2000=r"llamaindex_entities_0.2"
documentsPath_2000=r"Text_Files_Old"
index_2000=indexgenerator(indexPath_2000,documentsPath_2000)
vector_retriever_2000 = VectorIndexRetriever(index=index_2000,similarity_top_k=2)
bm25_retriever_2000 = BM25Retriever.from_defaults(index=index_2000, similarity_top_k=2)
postprocessor = LongContextReorder()

class HybridRetriever(BaseRetriever):
    def __init__(self,vector_retriever_2000, bm25_retriever_2000):
        #self.vector_retriever_1000 = vector_retriever_1000
        #self.bm25_retriever_1000 = bm25_retriever_1000
        self.vector_retriever_2000 = vector_retriever_2000
        self.bm25_retriever_2000 = bm25_retriever_2000
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes_2000 = self.bm25_retriever_2000.retrieve(query, **kwargs)
        vector_nodes_2000 = self.vector_retriever_2000.retrieve(query, **kwargs)
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in bm25_nodes_2000+vector_nodes_2000])
        num_token =  len(encoding.encode(context_str))
        if num_token > 3900:
            all_nodes = postprocessor.postprocess_nodes(nodes=bm25_nodes_2000+vector_nodes_2000,query_bundle=QueryBundle(query_str=prompt.lower()))
        else:
            all_nodes = bm25_nodes_2000+vector_nodes_2000
        return all_nodes
hybrid_retriever=HybridRetriever(vector_retriever_2000,bm25_retriever_2000)

# User-provided prompt
page_bg_img = '''
<style>
body {
background-image: url("https://csrbox.org/media/Hero-Image.png");
background-size: cover;
}
</style>
'''
memory=ChatMemoryBuffer.from_defaults(token_limit=3900)
rouge = Rouge()

condense_prompt = (
  "Given the following conversation between a user and an AI assistant and a follow up question from user,"
  "rephrase the follow up question to be a standalone question.\n"
  "Chat History:\n"
  "{chat_history}"
  "\nFollow Up Input: {question}"
  "\nStandalone question:")

context_prompt=(
        "You are a helpful and friendly chatbot who addresses queries in detail regarding I-Venture @ ISB."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history above and context, to interact and help the user. Never give any kinds of links, email addresses or contact numbers in the answer."
        )
def get_response(prompt,message_history):
    llm = OpenAI(model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)
    #query_engine=RetrieverQueryEngine.from_args(retriever=hybrid_retriever,service_context=service_context,verbose=True)
    chat_engine=CondensePlusContextChatEngine.from_defaults(retriever=hybrid_retriever,service_context=service_context,chat_history=message_history,context_prompt=context_prompt,condense_prompt=condense_prompt)
    nodes = hybrid_retriever.retrieve(prompt.lower())
    response = chat_engine.chat(str(prompt.lower()))
    validating_prompt = ("""You are an intelligent bot designed to assist users on an organization's website by answering their queries. You'll be given a user's question and an associated answer. Your task is to determine if the provided answer effectively resolves the query. If the answer is unsatisfactory, return 0.\n
                        Query: {question}  
                        Answer: {answer}
                        Your Feedback:
                        """)
    feedback = llm.complete(validating_prompt.format(question=prompt,answer=response.response))
    if feedback.text==str(0):
        st.write("DISTANCE APPROACH")
        response , joined_text=answer_question(prompt.lower())
        scores = rouge.get_scores(response, joined_text)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
        message_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=str(response)),)
        response_list = [response, prompt , scores]  
        df = pd.read_csv('logs/conversation_log.csv')
        new_row = {'Question': str(prompt), 'Answer': response,'Unigram_Recall' : scores[0]["rouge-1"]["r"],'Unigram_Precision' : scores[0]["rouge-1"]["p"],'Bigram_Recall' : scores[0]["rouge-2"]["r"],'Bigram_Precision' : scores[0]["rouge-2"]["r"]}
        df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        df.to_csv('logs/conversation_logs.csv', index=False)
        bucket = 'aiex' # already created on S3
        csv_buffer = StringIO()
        df.to_csv(csv_buffer)
        s3_resource= boto3.resource('s3',aws_access_key_id=os.environ['ACCESS_ID'],aws_secret_access_key= os.environ['ACCESS_KEY'])
        s3_resource.Object(bucket, 'conversation_log.csv').put(Body=csv_buffer.getvalue())  
        return response_list                                 
    else:
        context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes])
        scores=rouge.get_scores(response.response,context_str)
        message = {"role": "assistant", "content": response.response}
        st.session_state.messages.append(message)
        message_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=str(response.response)),)
        response_list = [response.response , prompt , scores]
        df = pd.read_csv('logs/conversation_logs.csv')
        new_row = {'Question': str(prompt), 'Answer': response.response,'Unigram_Recall' : scores[0]["rouge-1"]["r"],'Unigram_Precision' : scores[0]["rouge-1"]["p"],'Bigram_Recall' : scores[0]["rouge-2"]["r"],'Bigram_Precision' : scores[0]["rouge-2"]["r"]}
        df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        df.to_csv('logs/conversation_logs.csv', index=False)
        bucket = 'aiex' # already created on S3
        csv_buffer = StringIO()
        df.to_csv(csv_buffer)
        s3_resource= boto3.resource('s3',aws_access_key_id=os.environ['ACCESS_ID'],aws_secret_access_key=os.environ['ACCESS_KEY'])
        s3_resource.Object(bucket, 'conversation_log.csv').put(Body=csv_buffer.getvalue())
        return response_list 

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [{"role": "assistant", "content": "Ask anything about I-Venture @ ISB!"}]
if "message_history" not in st.session_state.keys():
    st.session_state.message_history=[ChatMessage(role=MessageRole.ASSISTANT,content="Ask anything aboug I-Venture @ ISB."),]
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.message_history.append(ChatMessage(role=MessageRole.USER,content=str(prompt)))

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(prompt=prompt,message_history=st.session_state.message_history)
                st.write(response[0])



        


                        
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
