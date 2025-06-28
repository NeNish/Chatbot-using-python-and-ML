# Named Entity Recognition (NER) using BiLSTM-LSTM

This repository demonstrates the implementation of **Named Entity Recognition (NER)** using a **BiLSTM-LSTM model** with Keras. NER is a key task in Natural Language Processing (NLP), where the goal is to identify and classify entities in text such as persons, organizations, locations, etc.

---

## 📌 What is Named Entity Recognition?

Named Entity Recognition means identifying and classifying real-world objects from text — such as:

- **Persons** (e.g., "Neha")
- **Locations** (e.g., "India")
- **Organizations** (e.g., "Google")
- **Professions or Fields** (e.g., "Machine Learning", "Trainer")

**Example**:  
In the sentence:  
`"My name is Nishant, and I am a Machine Learning Intern."`  
NER identifies:
- Nishant → Person  
- Machine Learning → Field  
- Intern → Profession

---

## 📂 Dataset

The dataset used is an annotated corpus in CSV format. You can download it from [Kaggle](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus).

---

## 🚀 Step-by-Step Code (All Embedded)

### 1. 📥 Load Data

```python
from google.colab import files
uploaded = files.upload()

import pandas as pd
data = pd.read_csv('ner_dataset.csv', encoding='unicode_escape')
data.head()

2. 🧹 Data Preparation
Map Words and Tags to Indices
def get_dict_map(data, token_or_tag):
    if token_or_tag == 'token':
        vocab = list(set(data['Word'].to_list()))
    else:
        vocab = list(set(data['Tag'].to_list()))
    
    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok

token2idx, idx2token = get_dict_map(data, 'token')
tag2idx, idx2tag = get_dict_map(data, 'tag')

data['Word_idx'] = data['Word'].map(token2idx)
data['Tag_idx'] = data['Tag'].map(tag2idx)
data_fillna = data.fillna(method='ffill', axis=0)

data_group = data_fillna.groupby(['Sentence #'], as_index=False)['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))
3. 🔀 Split Data
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def get_pad_train_test_val(data_group, data):
    n_token = len(set(data['Word'].to_list()))
    n_tag = len(set(data['Tag'].to_list()))

    tokens = data_group['Word_idx'].tolist()
    maxlen = max([len(s) for s in tokens])
    pad_tokens = pad_sequences(tokens, maxlen=maxlen, padding='post', value=n_token - 1)

    tags = data_group['Tag_idx'].tolist()
    pad_tags = pad_sequences(tags, maxlen=maxlen, padding='post', value=tag2idx["O"])
    pad_tags = [to_categorical(i, num_classes=n_tag) for i in pad_tags]

    tokens_, test_tokens, tags_, test_tags = train_test_split(pad_tokens, pad_tags, test_size=0.1, random_state=2020)
    train_tokens, val_tokens, train_tags, val_tags = train_test_split(tokens_, tags_, test_size=0.25, random_state=2020)

    return train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags

train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = get_pad_train_test_val(data_group, data)
4. 🧠 Build the BiLSTM-LSTM Model
import numpy as np
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

input_dim = len(set(data['Word'].to_list())) + 1
output_dim = 64
input_length = max([len(s) for s in data_group['Word_idx'].tolist()])
n_tags = len(tag2idx)

def get_bilstm_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(TimeDistributed(Dense(n_tags, activation="relu")))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
5. 🏋️ Train the Model
def train_model(X, y, model):
    loss = []
    for i in range(25):
        hist = model.fit(X, y, batch_size=1000, epochs=1, validation_split=0.2, verbose=1)
        loss.append(hist.history['loss'][0])
    return loss

model_bilstm_lstm = get_bilstm_lstm_model()
results = train_model(train_tokens, np.array(train_tags), model_bilstm_lstm)
6. ✅ Test with spaCy (Optional)
You can also try a built-in model like spaCy to see how well NER works:


import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
text = nlp("Hi, My name is Aman Kharwal. I am from India. I want to work with Google. Steve Jobs is my inspiration.")
displacy.render(text, style='ent', jupyter=True)
✅ Output
The trained model can recognize and label entities in the input text, such as:

Person: Neha Reddy, Steve Jobs

Location: India

Organization: Google

🧾 Requirements
Install the necessary libraries:
pip install pandas numpy tensorflow keras scikit-learn spacy
python -m spacy download en_core_web_sm

📁 File Structure

├── ner_dataset.csv           # Input dataset
├── ner_model.py              # Complete training script (optional)
├── README.md                 # This file
🙋 Author
E S Nishant
⭐️ Show Support
If you find this helpful, consider giving a ⭐ on GitHub!












