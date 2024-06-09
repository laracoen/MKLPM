import numpy as np
import os
import gradio as gr
import xgboost as xgb
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


os.environ["WANDB_DISABLED"] = "true"

label2id =  {
    0: "mješovito",
    1: "negativno",
    2: "neutralno",
    3: "pozitivno"
  }
# names of the files saved in step 2: Training

model_file_name = "model.pkl"
vectorizer_file_name = 'vectorizer.pk'


# load
xgb_model_loaded = pickle.load(open(model_file_name, "rb"))
vectorizer_loaded = pickle.load(open(vectorizer_file_name, "rb"))


def predict_sentiment(predict_texts):
    predictions_loaded = xgb_model_loaded.predict(vectorizer_loaded.transform([predict_texts]))
    print(predictions_loaded)
    return label2id[predictions_loaded[0]]


interface = gr.Interface(
    fn=predict_sentiment,
    inputs='text',
    outputs=['text'],
    title='Croatian Book reviews Sentiment Analysis',
    examples= ["Volim kavu","Ne volim kavu"],
    description='Get the positive/neutral/negative sentiment for the given input.'
)

interface.launch(inline = False)   