from qa_cc import chat_gpt
from rouge import Rouge
import time

def get_reference_response(prompt_input):
    reference_response = chat_gpt(prompt_input)[0]
    return reference_response

def calculate_rouge_scores(answer, reference_response):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(answer, reference_response)
    return rouge_scores

def evaluate_response_time(func, *args, **kwargs):
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    response_time = end_time - start_time
    return response_time
