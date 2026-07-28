import re
import pickle
import keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

TOKENIZER_PATH = "models/tokenizer.pickle"
LSTM_MODEL_PATH = "models/lstm.keras"
MAXLEN = 150


with open(TOKENIZER_PATH, "rb") as f:
   tokenizer = pickle.load(f)

model = keras.models.load_model(LSTM_MODEL_PATH)

stop_words = stopwords.words('english')

def clean_txt(text):
   """Clean the review"""
   text = text.lower()
   cleaned_text = re.sub(r'[^a-z0-9\s]', '', text)
   tokens = word_tokenize(cleaned_text)
   filtered_text = [word for word in tokens if word not in stop_words]
   filtered_text = ' '.join(filtered_text)
   return filtered_text

def predict_sentiment(review):
  """Predicts the sentiment of the review"""
  text = clean_txt(review)
  seq = tokenizer.texts_to_sequences([text])
  padded = pad_sequences(seq, maxlen= MAXLEN, dtype='float32', padding='pre', truncating='pre')
  pred = model.predict(padded)

  if pred >= 0.5:
    return "Positive"
  else:
    return "Negative"
     

