import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


bow_tf_idf_vectorizer :dict = None

with open("data/bow_tfidf_vectorizer.pkl", "rb") as pickle_file:
    bow_tf_idf_vectorizer = pickle.load(pickle_file)


def convert_into_bow(df):
    count_vectorizer :CountVectorizer = bow_tf_idf_vectorizer['bow_count_vect']
    count_vectorized_data = count_vectorizer.transform(df)
    # print(f"The data type of the bow data: {type(count_vectorized_data)}")
    return count_vectorized_data.toarray()


def convert_into_tf_idf(df):
    tfidf_vectorizer :TfidfVectorizer = bow_tf_idf_vectorizer['tfidf_vect']
    tf_idf_data = tfidf_vectorizer.transform(df)
    # print(f"The data type of the tfidf data: {type(tf_idf_data)}")
    return tf_idf_data.toarray()

