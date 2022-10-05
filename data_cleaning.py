# import
import re
import nltk
nltk.data.path.append('/app/nltk_data')
from nltk.stem import WordNetLemmatizer  # lemmatizer
from nltk.stem import PorterStemmer
# nltk.download('stopwords')
from nltk.corpus import stopwords


'''
cleans the data.
removes the punctuation
removes the extra special characters.
'''

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

print(f"Length of the stop words: {len(stop_words)}")

def clean_data(text):
    text = ' '.join(e for e in text.split() if e.lower() not in stopwords.words('english') and len(e)>2) 
    # convert to lower and remove stopwords discard words whose len < 2
    text = re.sub(r"won\'t", "", text) # decontracting the words
    text = re.sub(r"can\'t", "", text)
    text = re.sub(r"n\'t", "", text)
    text = re.sub(r"\'re", "", text)
    text = re.sub(r"\'s", "", text)
    text = re.sub(r"\'d", "", text)
    text = re.sub(r"\'ll", "", text)
    text = re.sub(r"\'t", "", text)
    text = re.sub(r"\'ve", "", text)
    text = re.sub(r"\'m", "", text)
    text = re.sub(r'\W', ' ', str(text))  # Remove all the special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # remove all single characters 
    text = re.sub(r"[0-9]", " ", text)  # replace number with space
    text = re.sub(r"[^A-Za-z_]", " ", text)  # replace all the words except "A-Za-z_" with space
    text = re.sub(r'[^\w\s]','',text)
    text = text.lower().strip()   # strip

    # Lemmatization
    tokens = text.split()
    
    tokens_res = []
    
    for word in tokens:
        word = lemmatizer.lemmatize(word)
        word = stemmer.stem(word)
        if word not in stop_words and len(word) > 2:
            tokens_res.append(word)
    
    new_text = ' '.join(tokens_res)
    
    return new_text