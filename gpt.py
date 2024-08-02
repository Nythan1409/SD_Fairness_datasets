import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import openai

PATH = "./formatted_datasets/"
NEW_PATH = "./predicted_datasets/"

CLIENT = openai.OpenAI(
  api_key='***',
  project='***'
)

def predict_stance_gpt(text:str, target:str, keyword:str=None):
    if keyword:
        text = "I am "+keyword+". "+text
    task = "Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target.\nAnalyze the following social media statement and determine its stance towards the provided [target]. Respond with a single word: \"FAVOR\" or \"AGAINST\". Only return the stance as a single word, and no other text.\n\n[target]: "+target+"\nStatement: "+text
    output = CLIENT.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "user", "content": task},
                        ]
            )
    answer = output.choices[0].message.content
    result = ''.join(filter(str.isalnum, answer)).lower()
    if "against" in result and not ("favor" in result):
        result = -1
    elif "favor" in result and not ("against" in result):
        result =  1
    else:
        result =  0
    return result

def prediction_gpt(dataset:pd.DataFrame, keyword:str=None):
    new_dataset = dataset[["Text", "Stance", "Class", "Target"]].copy()
    new_dataset.insert(4, "Prediction", new_dataset.apply(lambda x: predict_stance_gpt(x["Text"], x["Target"], keyword), axis=1))
    return new_dataset

def run_gpt(dataset:pd.DataFrame, batch_size:int, save_as:str="results.csv", keyword:str=None):
    size = dataset.shape[0]
    new_datas = []
    for i in tqdm(range(0, size, batch_size)):
        
        new_data = prediction_gpt(dataset.iloc[i:np.min([i+batch_size, size]), :], keyword)

        new_datas.append(new_data)

        print(f"{(i+1)*batch_size} done")

    pd.concat(new_datas, ignore_index=True).to_csv(NEW_PATH + save_as)

varieties = pd.read_csv(PATH+"varieties.csv")
readability = pd.read_csv(PATH + "scd_readability.csv")
new_readability = pd.read_csv(PATH + "kemlm_readability.csv")
# full_data = pd.concat([varieties, readability])
both_readability = pd.concat([readability, new_readability])



run_gpt(varieties, 100, "results_gpt_on_varieties.csv")