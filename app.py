from pyexpat import model
from tkinter.messagebox import NO
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

from data_cleaning import clean_data
from featurize_vector import convert_into_bow, convert_into_tf_idf, bow_tf_idf_vectorizer
from data_preprocessing import perform_min_max_scaling, min_max_scaling
from model import get_model


file = st.file_uploader("Upload the safe city raw file", type=['csv'])
input_text = st.text_input("Enter the text to predict")

vectorizer = st.radio("Select your feature vectorizer", ['Bag of words','TF Idf Vectorizer'])
model_type = st.radio("Select your model", ['Random Forest','XGBoost','Keras'])


def pre_process(df :pd.DataFrame, vectorizer):
    desc = df['Description']
    
    if vectorizer=='Bag of words':
        data = convert_into_bow(desc)
    else:
        data = convert_into_tf_idf(desc)
        data = perform_min_max_scaling(data)
    
    print(f"The shape of the input vectorized data : {data.shape} from the vectorizer: {vectorizer}")

    return data


def data_cleaning(df :pd.DataFrame):
    df['Description'] = df['Description'].map(lambda a: clean_data(a))


def predict(data, vectorizer, model_type):
    model = get_model(vectorizer, model_type)
    result = None
    if model is not None:
        result = model.predict(data)
        print(f"Model prediction is done...")
    else:
        st.error("Selected combination is invalid...")

    return result

def post_process(input_description, result, model_type):
    
    if result is not None:
        df_result = pd.DataFrame(result, columns=['Commenting','Ogling','Touching'])
        df_result['input_desc'] = input_description
        if model_type == 'Keras':
            for column in ['Commenting','Ogling','Touching']:
                df_result[column] = df_result[column].map(lambda val: 1 if val >=0.5 else 0)
        st.table(df_result)

def run():
    if model_type=='Keras' and vectorizer == 'Bag of words':
            st.error("INVALID feature and model selection. Try different combination")
    else:
        if file is not None:
            bytes_data = file.getvalue()
            s = str(bytes_data,'utf-8')
            data = StringIO(s) 
            df=pd.read_csv(data)
            input_description = df['Description'].copy()
            data_cleaning(df)
            data = pre_process(df, vectorizer)
            result = predict(data, vectorizer, model_type)
            post_process(input_description, result, model_type)
        elif input_text is not None:
            print(f"The input text is being predicted.....")
            input_value = clean_data(input_text)
            if vectorizer == "Bag of words":
                input_value = convert_into_bow([input_value])
            else:
                input_value = convert_into_tf_idf([input_value])
                input_value = min_max_scaling.transform(input_value)
            
            model = get_model(vectorizer, model_type)
            result = model.predict(input_value)[0]
            print(f"Result: {result}")
            if model_type == 'Keras':
                result_dict = {
                    'input_text': input_text,
                    'Commenting': 1 if result[0] >= 0.5 else 0,
                    'Ogling': 1 if result[1] >= 0.5 else 0,
                    'Touching':1 if result[2] >= 0.5 else 0,
                }
            else:
                result_dict = {
                    'input_text': input_text,
                    'Commenting': result[0],
                    'Ogling': result[1],
                    'Touching':result[2],
                }

            st.json(result_dict)

        else:
            st.error("The input file is empty")



predict_button = st.button("Predict", on_click=run)
