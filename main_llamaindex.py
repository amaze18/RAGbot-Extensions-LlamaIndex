#-------------STD DEPENDENCIES---------------------------#
import os
from collections import namedtuple
import math
import time
from datetime import timedelta

#---------------CHATBOT DEPENDENCIES-----------------#
import altair as alt
import pandas as pd
import openai
#from hugchat import hugchat
#from hugchat.login import Login
#import openpyxl
from llama_index import StorageContext, load_index_from_storage

#----------------------UI DEPENDENCIES---------------#
from PIL import Image

import streamlit as st

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


SECRET_TOKEN = os.environ["SECRET_TOKEN"]
openai.api_key = SECRET_TOKEN

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



storage_context = StorageContext.from_defaults(persist_dir="BITSPilani/")
index = load_index_from_storage(storage_context)

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
# User-provided prompt
page_bg_img = '''
<style>
body {
background-image: url("https://csrbox.org/media/Hero-Image.png");
background-size: cover;
}
</style>
'''

#st.markdown(page_bg_img, unsafe_allow_html=True)
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            starttime=time.perf_counter()
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            duration=timedelta(seconds=time.perf_counter()-starttime)
            st.write(response.response)
            st.write("Response Time: "+str(duration))
            """
            workbook = openpyxl.load_workbook(r"C:\Users\Kush Juvekar\Desktop\Bot_Answers.xlsx")
            sheet = workbook.active
            data=[[str(answer),str(duration)]]
            for row in data:
                sheet.append(row)
            workbook.save(r"C:\Users\Kush Juvekar\Desktop\Bot_Answers.xlsx")
            """
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
