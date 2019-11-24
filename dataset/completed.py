#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import tkinter as tk
from tkinter import filedialog
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# import numpy as np 
#import pandas as pd 
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras

print("You have TensorFlow version", tf.__version__)


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = re.sub(r'\d+', '', word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words



def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    return words

def preprocess(sample):
    sample = remove_URL(sample)
    sample = replace_contractions(sample)
    # Tokenize
    words = nltk.word_tokenize(sample)

    # Normalize
    return normalize(words)
# In[3]:


data = pd.read_csv("F:/proteger_system/file_classification/dataset/all.csv")  


# In[4]:


data['category'][0]


# In[5]:


data['text'][0]


# In[6]:


max_words = 10000
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words)


# In[7]:


tokenize.fit_on_texts(data['text']) # fit tokenizer to our training text data


# In[8]:


# x_train = tokenize.texts_to_matrix(data['text'])


# In[9]:


tokenize.word_index


# In[ ]:





# In[10]:


# tokenize.fit_on_sequences(data['text'])


# In[11]:


# tokenize.texts_to_sequences


# In[12]:


x_data = tokenize.texts_to_sequences(data['text'])


# In[13]:


print(x_data[0])


# In[14]:


encoder = LabelEncoder()
encoder.fit(data['category']) 
y_data = encoder.transform(data['category'])


# In[ ]:





# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


train_data, test_data, train_labels, test_labels = train_test_split(x_data, y_data, test_size=0.33, random_state=42)


# In[17]:


print(f"Training entries {len(train_data)}. Labels: {len(train_labels)}")


# In[18]:


print(f"Testing entries {len(test_data)}. Labels: {len(test_labels)}")


# In[19]:


len(train_data[0])


# In[20]:


# tokenize.sequences_to_texts(sequences)


# In[21]:


print(train_labels[1])


# In[22]:


word_index = tokenize.word_index


# In[23]:


word_index = {k : (v+3) for k, v in word_index.items()}


# In[24]:


word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


# In[25]:


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# In[26]:


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, "?") for i in text])


# In[27]:


print(decode_review(train_data[0]))


# In[28]:


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding="post",
                                                        maxlen=1024)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding="post",
                                                       maxlen=1024)


# In[29]:


print(len(train_data[0]), len(train_data[1]))

print(train_data[0])

vocab_size = 10000


# In[30]:


from keras.utils import to_categorical
org_test_labels = test_labels
org_train_labels = train_labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[31]:


model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(3, activation=tf.nn.softmax))


# In[32]:


model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["acc"])


# In[33]:


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# In[34]:


history = model.fit(train_data, train_labels,
                    batch_size=512,
                    epochs=5,
                    verbose=1,
                    validation_split=0.1)


# In[35]:


# model.fit(partial_x_train,partial_y_train,epochs=40,batch_size=512,verbose=1)


# In[36]:


results = model.evaluate(test_data, test_labels)

print(results)


# In[37]:


index = 0
test_review = test_data[index]
predict = model.predict_classes([test_review])
print("Text: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: "+str(org_test_labels[index]))


# In[ ]:
import PyPDF2


if __name__ == "__main__":
    root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
print(file_path)

base=os.path.basename(file_path)
F_name = os.path.splitext(os.path.basename(file_path))[0]
pdfFileObj = open (file_path, 'rb')
pdfReader = PyPDF2.PdfFileReader(
        pdfFileObj)
pagenum = pdfReader.numPages
count = 0
text1 = ""

word2 = ""
while count < pagenum:
    pageObj = pdfReader.getPage(count)
    count+=1
    text1 +=pageObj.extractText()


    sample = text1              
    




# In[38]:



input_text = sample
input_seq = tokenize.texts_to_sequences(input_text)

input_seq = keras.preprocessing.sequence.pad_sequences(input_seq,
                                                       value=word_index["<PAD>"],
                                                       padding="post",
                                                       truncating='post',
                                                       maxlen=1024)
#print(decode_review(input_seq[0]))
predict = model.predict_classes([input_seq])
print("Prediction: " + str(predict[0]))
print("Prediction class name: " + str(encoder.inverse_transform([predict[0],0])[0]))
predictionn = str(encoder.inverse_transform([predict[0],0])[0])
print(predictionn)




import shutil, os


# In[ ]:
if(predictionn == 'public'):
    shutil.copy(file_path, 'F:/proteger_system/dmtool/kanishka/temp')
    os.rename('F:/proteger_system/dmtool/kanishka/temp/'+F_name+".pdf",'F:/proteger_system/dmtool/kanishka/temp/'+F_name+".public.pdf")   
    
   
if(predictionn == 'internal'):
    shutil.copy(file_path, 'F:/proteger_system/dmtool/kanishka/temp')
    os.rename('F:/proteger_system/dmtool/kanishka/temp/'+F_name+".pdf",'F:/proteger_system/dmtool/kanishka/temp/'+F_name+".internal.pdf")   
    

if(predictionn == 'confidential'):
    shutil.copy(file_path, 'F:/proteger_system/dmtool/kanishka/temp')
    os.rename('F:/proteger_system/dmtool/kanishka/temp/'+F_name+".pdf",'F:/proteger_system/dmtool/kanishka/temp/'+F_name+".confidential.pdf")   
    

