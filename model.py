import pickle
from tensorflow import keras
import xgboost

with open("data\Ensemble_models/Random_forest_bow.pkl", "rb") as pickle_file:
    Random_forest_bow = pickle.load(pickle_file)

with open("data\Ensemble_models/Random_forest_tfidf.pkl", "rb") as pickle_file:
    Random_forest_tfidf = pickle.load(pickle_file)

with open("D:\Main thesis\safecity-master\SafeCity_Main Project\Best_model/xgboost_bow.pkl", "rb") as pickle_file:
    xgboost_bow = pickle.load(pickle_file)

with open("data\\Ensemble_models\\xgboost_tfidf.pkl", "rb") as pickle_file:
    xgboost_tfidf = pickle.load(pickle_file)

keras_tfidf = keras.models.load_model('data/keras_tfidf/')

if keras_tfidf:
    print("Keras Model is loaded")


'''
Get the model based on the user selection
'''
def get_model(vectorizer, model_type):
    
    print(f"The vectorizer {vectorizer} and model type: {model_type}")

    if vectorizer=="Bag of words" and model_type=="XGBoost":
        model = xgboost_bow
    elif vectorizer=="Bag of words" and model_type=="Random Forest":
        model = Random_forest_bow
    elif vectorizer=="TF Idf Vectorizer" and model_type=="XGBoost":
        model = xgboost_tfidf
    elif vectorizer=="TF Idf Vectorizer" and model_type=="Random Forest":
        model = Random_forest_tfidf
    elif vectorizer=="TF Idf Vectorizer" and model_type=="Keras":
        model = keras_tfidf
    else:
        model = None
    
    return model
