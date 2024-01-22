import pandas as pd
import os
import tiktoken
import numpy as np
import openai
key=os.environ['SECRET_TOKEN']
openai.api_key=key
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=key)
from numpy  import reshape
from openai import OpenAI
client = OpenAI(api_key=os.environ['SECRET_TOKEN'])
from ast import literal_eval
from chatbot_utils import distances_from_embeddings, cosine_similarity
"""
# Assuming all your text files are in a directory named 'text_files'
directory_path = 'scraped_files\processed\striped_files_new'
files = [file for file in os.listdir(directory_path) if file.endswith('.txt')]

data = {'title': [], 'text': []}

for file_name in files:
    file_path = os.path.join(directory_path, file_name)
    
    # Assuming each line in the text file is a separate record
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            text = file.read()

    
    data['title'].append(file_name)
    data['text'].append(text)

# Create a DataFrame
df = pd.DataFrame(data)

tokenizer = tiktoken.get_encoding("cl100k_base")
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()
max_tokens = 500

def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks


def shorten(df):
    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )

    new_df = pd.DataFrame(shortened, columns = ['text'])
    new_df['n_tokens'] = new_df.text.apply(lambda x: len(tokenizer.encode(x)))
    return new_df

df = shorten(df)
df.n_tokens.hist()
df['embeddings'] = df.text.apply(lambda x: client.embeddings.create(input=x, model='text-embedding-ada-002').data[0].embedding)
df.to_csv('embeddings_new.csv')

from ast import literal_eval

df = pd.read_csv('embeddings_new.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
"""
def create_context(
    question, df=pd.read_csv('embeddings_new.csv',index_col=0)
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    max_len=1800
    size="ada"
    # Get the embeddings for the question
    # print("question::",question)
    #st.write(question)
    #q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    q_embeddings=client.embeddings.create(input = question, model="text-embedding-ada-002").data[0].embedding
    # Get the distances from the embeddings
    df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(question,):
    
 
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    model="gpt-3.5-turbo"
    max_len=1800
    size="ada"
    debug=False
    max_tokens=250
    stop_sequence=None
    context = create_context(question,df=pd.read_csv('embeddings_new.csv',index_col=0))
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    introduction = 'Use the below text to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question_ai = f"\n\nQuestion: {question}"
    message = introduction
    message = message + context + question_ai
    messages = [
        {"role": "system","content": "You are iVBot, an AI based chatbot assistant. You are friendly, proactive, factual and helpful, \
        you answer from the context provided"}, {"role": "user", "content": message},
    ]
    
    try:
        response = client.chat.completions.create(
         model='gpt-3.5-turbo',
        messages=messages,
         temperature=0.01,
          top_p=0.75,
         
        )
      
        ans=response.choices[0].message.content
        # Create a completions using the questin and context
        
        #response = openai.Completion.create(
         #   prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I do not know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
         #   temperature=0.08,
         #   max_tokens=max_tokens,
         #   top_p=0.75,
         #   frequency_penalty=0,
         #   presence_penalty=0,
         #   stop=stop_sequence,
         #   model=model,
        #)
      
        return ans #response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


def generate_response(prompt_input, email, passwd):
     question0=prompt_input
     question=prompt_input
     ans = answer_question(prompt_input)
     # st.write(ans)
     if (ans=='I don\'t know.' or ans=='I don\'t know' or ans== 'I could not find an answer.' or 'I could not find' in ans  or ' I couldn\'t find'  in ans  ):
           print(f"{question}"+"  FAILED!")
           print(ans)
           question=question0+ " ISB DLabs"
           ans=answer_question(question)
           print(ans)
           if (ans=='I don\'t know.'  or ans=='I don\'t know' or ans== 'I could not find an answer.' or 'I could not find' in ans or ' I couldn\'t find'  in ans  ):
             print(f"{question}"+"  FAILED!")
             question=question0+ " ISB"
             ans=answer_question(question)
             print(ans)
           else:
             print(f"{question} WORKED")
             if (ans=='I don\'t know.'  or ans=='I don\'t know'  or ans== 'I could not find an answer.' or 'I could not find' in ans or ' I couldn\'t find'  in ans  ):
               print(f"{question}"+"  FAILED!")
               question=question0+ " I-Venture @ ISB"
               ans=answer_question(question)
               if (ans=='I don\'t know.'  or ans=='I don\'t know'  or ans== 'I could not find an answer.' or 'I could not find' in ans or ' I couldn\'t find'  in ans  ):
                   print(f"{question}"+"  FAILED!")
                   question=question0+ "Dlabs ISB"
                   ans=answer_question(question)
                   if (ans=='I don\'t know.'  or ans=='I don\'t know'  or ans== 'I could not find an answer.' or 'I could not find' in ans or ' I couldn\'t find'  in ans  ):
                       print(f"{question}"+"  FAILED!")
                       question=question0+ "Indian School of Business"
                       ans=answer_question(question)
     return ans
