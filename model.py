from embedding import Preprocessing_Embedding
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf

def fit_embedding(path='clean_data.csv'):
	""""Input : Path to the clean dataset
	Outputs : Embedding matrix, Fasttext model, vocabulary size, embedding_dim,df_question,df_answer"""
	df=pd.read_csv(path)
	df=df.fillna('no')
	pe=Preprocessing_Embedding (df,['question','answer'],10, load=False)
	embedding_matrix,ft_model,vocabulary_size,sequence_length, embedding_dim,df_encoded=pe.instantiateEmbenddingMatrix()
	df_question,df_answer=df_encoded
	return embedding_matrix,ft_model,vocabulary_size,sequence_length, embedding_dim,df_question,df_answer
	
def create_train_test_val(X,y):
	"""Input : X and y
	Outputs : Train,test,val dataset"""
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
	x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
	return x_train,y_train,x_test,y_test,x_val,y_val
	
def build_model(vocabulary_size, embedding_dim,sequence_length,embedding_matrix, rnn_units):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, 
                                            input_length=sequence_length,weights=[embedding_matrix],
                                            trainable=True))
  model.add(tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        recurrent_initializer='glorot_uniform',dropout=0.1))
  model.add(tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        recurrent_initializer='glorot_uniform',dropout=0.1))
  model.add(tf.keras.layers.Dense(vocabulary_size))
  return model
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
	
def fit_model(x_train,y_train,x_test,y_test,vocabulary_size, embedding_dim,sequence_length,embedding_matrix, rnn_units= 1024):
	"""Input : X and y
	Outputs : TF model"""
	model = build_model(
	  vocabulary_size = vocabulary_size,
	  embedding_dim=embedding_dim,
	  sequence_length=sequence_length,
	  embedding_matrix=embedding_matrix,
	  rnn_units=rnn_units)
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=loss)
	EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
	                              patience=2, min_lr=0.00001)
	model.fit(x=x_train,y=y_train, epochs=10, batch_size=64,validation_data=[x_test,y_test],callbacks=[EarlyStopping,reduce_lr])
	return model

def predict(model,x_val):
	"""Input : Tf model and a val dataset
	Outputs : prediction"""
	return model.predict(x_val)

	
def main():
 
  print('fit embedding...\n')

  embedding_matrix,ft_model,vocabulary_size,sequence_length, embedding_dim,df_question,df_answer=fit_embedding(path='clean_data.csv')

  print('Creating Test Train Val dataset...')
  x_train,y_train,x_test,y_test,x_val,y_val=create_train_test_val(df_question,df_answer)

  print('Building model...\n ')
  model=fit_model(x_train,y_train,x_test,y_test,vocabulary_size, embedding_dim,sequence_length,embedding_matrix, rnn_units= 1024)
  print('Predict model...\n ')
  predict(model,x_val)     

if __name__ == '__main__':
  main()