import pandas as pd
from transformers import pipeline
import torch
import os
from transformers.utils import logging
logging.set_verbosity_error()
import numpy as np

PATH = "./formatted_datasets/"
NEW_PATH = "./predicted_datasets/"



def predict_stance_ca(pipe, text:str, target:str, keyword:str=None):
    if keyword:
        text = "I am "+keyword+". "+text
    task = "Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target.\nAnalyze the following social media statement and determine its stance towards the provided [target]. Respond with a single word: \"FAVOR\" or \"AGAINST\". Only return the stance as a single word, and no other text.\n\n[target]: "+target+"\nStatement: "+text
    output=pipe(task)
    true_output = output[0]["generated_text"][len(task):]
    return true_output
    
def get_stance(text:str):
    result = ''.join(filter(str.isalnum, text)).lower()
    if "against" in result and not ("favor" in result):
        result = -1
    elif "favor" in result and not ("against" in result):
        result =  1
    else:
        result =  0
    return result

def prediction(pipe, dataset:pd.DataFrame, keyword:str=None):
    new_dataset = dataset[["Text", "Stance", "Class", "Target"]].copy()
    new_dataset.insert(4, "Generated_text", new_dataset.apply(lambda x: predict_stance_ca(pipe, x["Text"], x["Target"], keyword), axis=1))
    new_dataset.insert(4, "Prediction", new_dataset.apply(lambda x: get_stance(x["Generated_text"]), axis=1))
    
    return new_dataset



PIPE = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", token="***", torch_dtype=torch.bfloat16, device_map='cuda', max_new_tokens=50)
print("Model loaded", flush=True)



def run(dataset:pd.DataFrame, batch_size:int, save_as:str="results.csv", keyword:str=None):
    size = dataset.shape[0]
    new_datas = []
    for i in range(0, size, batch_size):
        
        new_data = prediction(PIPE, dataset.iloc[i:np.min([i+batch_size, size]), :], keyword)

        new_datas.append(new_data)

        print(f"{i+batch_size} done", flush=True)

    pd.concat(new_datas, ignore_index=True).to_csv(NEW_PATH + save_as)



varieties = pd.read_csv(PATH+"varieties.csv")
readability = pd.read_csv(PATH + "scd_readability.csv")
new_readability = pd.read_csv(PATH + "kemlm_readability.csv")
# full_data = pd.concat([varieties, readability])
# both_readability = pd.concat([readability, new_readability])



run(varieties, 100, "results_llama_on_varieties.csv")
