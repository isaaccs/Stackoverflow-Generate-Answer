from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np
from gensim.models.fasttext import FastText
import pandas as pd
class Preprocessing_Embedding ():
  def __init__(self,df,list_text_feature,embedding_dim , window_context =2, min_word_count = 2, sample = 1e-3, sg=0, overwrite=False, load=True):
    self.df=df
    self.list_text_feature=list_text_feature
    self.df_corpus=df
    self.embedding_dim=embedding_dim
    self.window_context=window_context
    self.min_word_count=min_word_count
    self.sample=sample
    self.sg=sg
    self.overwrite=overwrite
    self.load=load

  def create_corpus(self,df,list_text_feature):
    """Input: df: a pandas DF
      list_text_feature : a list of all text features ['question','answer']
      Output a corpus"""
    s=pd.Series([])
    for f in list_text_feature:
      s=s.append(df[f], ignore_index=True)
    df_corpus=pd.DataFrame(s,columns=['text'])
    self.df_corpus=df_corpus
    return df_corpus

  def create_analyser(self,df,col,type_ngrams = 'words') :
    if type_ngrams == 'words' :
        k1 = 1
        k2 = 1
    elif type_ngrams == 'N_grams' :
        k1 = 1
        k2 = 3
    elif type_ngrams == 'Only_N_grams' :
        k1 = 2
        k2 = 3
    vectorizer = TfidfVectorizer(ngram_range=(k1,k2),lowercase =False,stop_words=None)
    vectorizer.fit(df[col])
    analyser = vectorizer.build_analyzer()
    return analyser

  def create_docs2(self,df,col,analyser) :
    new = df.apply(lambda x:analyser(x[col]), axis=1)
    return new 

  def fit(self) :
    """ Fit all the text feature and create the vocabulary dictionnary"""
    text='text'
    type_ngrams='words'
    df_corpus=self.create_corpus(self.df,self.list_text_feature)
    analyser=self.create_analyser(df_corpus,text,type_ngrams)
    text=self.create_docs2(df_corpus,text,analyser)

    # Tokenization du text
    tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False, filters='')
    num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, lower=False, filters='')
    tokenizer.fit_on_texts(text)
    text = tokenizer.texts_to_sequences(text)
    maxlen = len(max(text,key=len))
    text_corpus = tf.keras.preprocessing.sequence.pad_sequences(sequences=text, maxlen=maxlen)
    input_dim = int(np.max(text_corpus) + 1)
    sequence_length = text_corpus.shape[1]
    vocabulary_size = input_dim
    self.maxlen=maxlen
    self.analyser=analyser
    self.tokenizer=tokenizer
    self.input_dim=input_dim
    self.sequence_length=sequence_length
    self.vocabulary_size=vocabulary_size


  def transform(self):
    """ Transform text into a sequence"""
    df_list=[]
    for f in self.list_text_feature:
      globals()['df_{}'.format(f)]=self.create_docs2(self.df,f,self.analyser)
      globals()['df_{}'.format(f)] = self.tokenizer.texts_to_sequences(globals()['df_{}'.format(f)])
      globals()['df_{}'.format(f)] = tf.keras.preprocessing.sequence.pad_sequences(sequences=globals()['df_{}'.format(f)], maxlen=self.maxlen)
      df_list.append(globals()['df_{}'.format(f)])
    return tuple(df_list)

  def instantiateEmbenddingMatrix(self):
    """ Fit or load a Embedding Matrix"""
    embedding_dim = self.embedding_dim
    window_context = self.window_context 
    min_word_count = self.min_word_count
    sample = self.sample
    sg=self.sg
    overwrite=self.overwrite
    load=self.load
    self.fit()
    df_corpus=self.create_corpus(self.df,self.list_text_feature)
    sentences_ted=df_corpus['text']
    tokenizer=self.tokenizer
    vocabulary_size=self.vocabulary_size
    sequence_length=self.sequence_length

    if load == True:
        try:
            embedding_matrix = None
            print("Loading embedding matrix...")
            embedding_matrix = np.genfromtxt('embedding.csv', delimiter=',')
            ft_model = FastText.load("ft_model.model")
            
        except:
            embedding_matrix = None
            pass
    else:
        embedding_matrix = None
    if embedding_matrix is None or overwrite or load == False:
        ft_model = FastText(sentences_ted,min_n=0,max_n=3
                            , size=embedding_dim, window=window_context,min_count=min_word_count,sample=sample, sg=sg, iter=10)    

        ft_model.save("ft_model.model")
        print('Preparing embedding matrix...')
        words_not_found = []
        nb_words = vocabulary_size
        word_index = tokenizer.word_index
        embedding_matrix = np.zeros((nb_words, embedding_dim))
        for word, i in word_index.items():
            if i >= nb_words:
                continue
            embedding_vector = ft_model.wv.get_vector(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                words_not_found.append(word)
        print(embedding_matrix.shape)
        print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        np.savetxt('embedding.csv', embedding_matrix, delimiter=',')
    *df_encoded, =self.transform()
            
    return embedding_matrix,ft_model,vocabulary_size,sequence_length, embedding_dim,df_encoded