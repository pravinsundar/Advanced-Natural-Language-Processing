
# Loading the Libraries

import nltk
nltk.download()
nltk.download('wordnet')
nltk.download('stopwords')

import pandas as pd
import numpy as np
import string, os
import scipy as sp
import sklearn
import sys
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

!pip install gensim
!pip install markovify

import markovify
from gensim.models import ldamodel
import gensim.corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import pickle


from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku


from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

! pip install pydrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Loading the data

file_id = '11fppMBgXWADXE5JqJ373d5AecjY4AYhF'
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile('abcnews-date-text.csv')

df = pd.read_csv("abcnews-date-text.csv")
df.head()

# Subsetting the data to 10,000 headlines

df_text = df[['headline_text']]
np.random.seed(1000);
df_text = df_text.iloc[np.random.choice(len(df_text), 10000)]

# Checking the Polarity of the words

from textblob import TextBlob

senti=[]
df_text_eda = df_text[:]
df_text_eda = df_text_eda.reset_index(drop=True)

for i in range(10000):
    print(i)
    txt=df_text_eda.loc[i,'headline_text']
    data_formatted=TextBlob(txt.strip())
    #print(data_formatted)
    ##txt=data_formatted.loc[9,'headline_text']
    ##print(txt)
    print(data_formatted)
    print(data_formatted.sentiment.polarity)
    if(data_formatted.sentiment.polarity==0):
        sentiment_for_sent='Neutral'
    elif(data_formatted.sentiment.polarity>0):
        sentiment_for_sent='positive'
    else:
        sentiment_for_sent='Negative'

    senti.append([txt,data_formatted.sentiment.polarity,sentiment_for_sent,data_formatted.sentiment.subjectivity])

senti=pd.DataFrame(senti)
senti.columns=['Headline','Polarity','Sentiment','subjectivity']
senti['Sentiment'].value_counts().plot(kind='bar')

senti.hist(column="subjectivity")

df_text1 = df_text[:]

df_text1 = df_text1.astype('str')

# Removing Stop Words

for i in range(len(df_text1)):

    df_text1.iloc[i]['headline_text'] = [k for k in df_text1.iloc[i]['headline_text'].split(' ') if k not in stopwords.words()]

# Lemmatization

wordnet_lemmatizer = WordNetLemmatizer()

for i in range(len(df_text1)):
  df_text1.iloc[i]['headline_text'] = [wordnet_lemmatizer.lemmatize(k) for k in df_text1.iloc[i]['headline_text']]

# Pickle dump

pickle.dump(df_text1, open('data_text.dat', 'wb'))

# Getting the headlines for training
train_data = [tt[0] for tt in df_text1.iloc[0:].values]

num_topics = 10

# LDA

ss = gensim.corpora.Dictionary(train_data)
corpus = [ss.doc2bow(i) for i in train_data]
lda = ldamodel.LdaModel(corpus=corpus, id2word=ss, num_topics=num_topics)

def clean_text(ss):
    ss = "".join(i for i in ss if i not in string.punctuation).lower()
    ss = ss.encode("utf8").decode("ascii",'ignore')
    return ss

# A sample word cloud
!pip install wordcloud
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
stopwords=set(STOPWORDS)
def show_wordcloud(data,title=None):
    wc=WordCloud(background_color="black", max_words=10000,stopwords=STOPWORDS, max_font_size= 40)
    wc.generate(" ".join(data))
    fig=fig = plt.figure(figsize=[8,5], dpi=80)
    plt.axis('off')
    if title:
        fig.suptitle(title,fontsize=16)
        fig.subplots_adjust(top=1)
        plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=1,interpolation='bilinear')
        plt.show()

corpus = [clean_text(i) for i in df_text['headline_text']]
show_wordcloud(corpus,title="Wordcloud for ABC News Headlines")

# Generating Topics from LDA
def get_lda_topics(model, num):
    word_dict = {};
    for i in range(num):
        words = model.show_topic(i, topn = 20)
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words]
    return pd.DataFrame(word_dict)

lda = get_lda_topics(lda, num_topics)
lda

lda1 = pd.DataFrame(lda)

# NMF

# TFIDF transformation
train_data1 = [' '.join(i) for i in train_data]
v = CountVectorizer(analyzer='word', max_features=5000)
counts_v = v.fit_transform(train_data1)

t = TfidfTransformer(smooth_idf=False)
tfidf = t.fit_transform(counts_v)
tfidf_normalized = normalize(tfidf, norm='l1', axis=1)

model = NMF(n_components=num_topics, init='nndsvd')
model.fit(tfidf_normalized)

def get_nmf_topics(model, words):


    feat_names = vectorizer.get_feature_names()

    word_dict = {};
    for i in range(num_topics):


        ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[k] for k in ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words

    return pd.DataFrame(word_dict)

nmf = get_nmf_topics(model, 20)
nmf

nmf1 = pd.DataFrame(nmf)

# Markov Chain Model

df_text2 = df_text[:]
mcm_model1 = markovify.NewlineText(df_text2['headline_text'], state_size = 2)

for i in range(10):
    print(mcm_model1.make_sentence())


# Ensembling 3 Markov models

mcm_model1 = markovify.Text(df_text2['headline_text'], state_size = 2)
mcm_model2 = markovify.Text(df_text2['headline_text'], state_size = 2)
mcm_model3 = markovify.Text(df_text2['headline_text'], state_size = 2)
model_combo = markovify.combine([ mcm_model1, mcm_model2, mcm_model3 ], [ 1.5, 1.5, 1 ])

for i in range(5):
    print(model_combo.make_sentence())

# Part-Of-Speech tagging
import re
!pip install spacy
import spacy

!python -m spacy download en_core_web_lg

nlp = spacy.load('en_core_web_lg')

class POSifiedText(markovify.Text):
    def word_split(self, sentence):
        return ["::".join((i.orth_, i.pos_)) for i in nlp(sentence)]

    def word_join(self, words):
        sentence = " ".join(i.split("::")[0] for i in words)
        return sentence

mcm_model_pos = POSifiedText(df_text2['headline_text'], state_size = 2)

for i in range(5):
    print(mcm_model_pos.make_sentence())

# Looking at similarity between topics generated by LDA and NMF

for i in range(10):
  print("Topic {} in LDA".format(i+1))
  for j in range(10):
    tt = lda1.iloc[:,i]
    tt1 = nmf1.iloc[:,j]
    sentence1 = " ".join(k for k in tt)
    sentence2 = " ".join(k for k in tt1)
    tokens1 = nlp(sentence1)
    tokens2 = nlp(sentence2)
    sim_values = []
    dist_values = []
    for token1 in tokens1:
      for token2 in tokens2:
          #print(token1, token2, token1.similarity(token2))
          sim_values.append(token1.similarity(token2))
          t1 = token1.vector
          t2 = token2.vector
          dist = np.sqrt(np.sum((t1-t2)**2))
          dist_values.append(dist)
    print("Topic {} in NMF".format(j+1))
    #print("/n")
    print("Word mover distance is {}".format(np.mean(dist_values)))



# LSTM

corpus = [clean_text(i) for i in df_text['headline_text']]
corpus[:10]

# N-Gram generator

tt = Tokenizer()

def get_sequence_of_tokens(corpus):

    tt.fit_on_texts(corpus)
    words = len(tt.word_index) + 1

    seq = []
    for i in corpus:
        token_list = tt.texts_to_sequences([i])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            seq.append(n_gram_sequence)
    return seq, words

inp, words = get_sequence_of_tokens(corpus)
inp[:10]

# Creating Predictors and Labels

def sentences(inp_seq):
    max_len = max([len(x) for x in inp_seq])
    inp_seq = np.array(pad_sequences(inp_seq, maxlen=max_len, padding='pre'))

    predictors, label = inp_seq[:,:-1],inp_seq[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_len

predictors, label, max_len = sentences(inp)

# 3 Layers, 100 Units and 50 epocs

def create_model(max_len, words):
    input_len = max_len - 1
    model = Sequential()


    model.add(Embedding(words, 10, input_length=input_len))


    model.add(LSTM(100))


    model.add(Dense(words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

model = create_model(max_len, words)
model.summary()

model.fit(predictors, label, epochs=50, verbose=2)

# Generating headlines based on input and LSTM

def headline_generator(text, next_words, model, max_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        text += " "+output_word
    return text.title()

print (headline_generator("weather", 5, model, max_len))
print (headline_generator("police", 4, model, max_len))
print (headline_generator("interview", 4, model, max_len))
print (headline_generator("soccer", 4, model, max_len))
print (headline_generator("sports", 4, model, max_len))
print (headline_generator("science and technology", 5, model, max_len))
print (headline_generator("health", 5, model, max_len))
print (headline_generator("cricket", 5, model, max_len))
print (headline_generator("John Cartwright", 5, model, max_len))
print (headline_generator("Health Bill", 5, model, max_len))

## References
##    https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df
##    https://www.kaggle.com/nulldata/meaningful-random-headlines-by-markov-chain/notebook
##    https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms/data
##    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
##   https://github.com/jsvine/markovify
##    https://medium.com/@rahulvaish/textblob-and-sentiment-analysis-python-a687e9fabe96
