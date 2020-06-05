#!/Users/Isaac/opt/anaconda3/envs/TF-CPU/bin/python
import numpy as np
import pandas as pd
import requests # Getting Webpage content
from bs4 import BeautifulSoup as bs # Scraping webpages
from sklearn.preprocessing import MultiLabelBinarizer
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()


def Scrap_WebPage(url):
	"""Url from stackover flow
	Outputs: Question, Title, Tags, Answer """
  # Using requests module for downloading webpage content
  response1 = requests.get(url)
  
  # Getting status of the request
  # 200 status code means our request was successful
  # 404 status code means that the resource you were looking for was not found
  response1.status_code
  
  # Parsing html data using BeautifulSoup
  soup1 = bs(response1.content, 'html.parser')
  
  # body
  body1 = soup1.select_one('body')
  
  # printing the object type of body
  type(body1)
  ##title
  question_links = body1.select("h1 a.question-hyperlink")
  title = [i.text for i in question_links]
  
  ##Tags
  tags_divs = body1.select('div.post-taglist')
  a_tags_list = [i.select('a') for i in tags_divs]
  tags = []
  for a_group in a_tags_list:
      tags.append([a.text for a in a_group])
  
  ##Corps Info
  corps_links = body1.select("div.post-text")
  ##Question
  p_list = [i.select('p') for i in corps_links]
  question = []
  for p_group in p_list:
      question.append([p.text for p in p_group])

  
  ## Code text
  code_list = [i.select('pre') for i in corps_links]
  code_question = []
  for c_group in code_list:
      code_question.append([c.text for c in c_group])
  
  ##Answer Info
  answer_links = body1.select("div.answercell > div.post-text")
  ##Answer text
  p_list = [i.select('p') for i in answer_links]
  answer = []
  for p_group in p_list:
      answer.append([p.text for p in p_group])
  
  ## Code text
  code_list = [i.select('pre') for i in answer_links]
  code_answer = []
  for c_group in code_list:
      code_answer.append([c.text for c in c_group])
  title=title[0]
  
  try :
    tags=tags[0]
  except :
      tags=tags
  try :
    question=question[0][0]
  except :
      question=question[0]
  code_question=code_question[0]
  try :
    answer=answer[0][0]
  except :
      answer=np.nan
  try :
    code_answer=code_answer[0]
  except :
    code_answer=np.nan
  data={'title':title,
        'tags':tags,
        'question':  question,
        'code_question':code_question,
        'answer':answer,
        'code_answer':code_answer}
  return data
	
  

def get_links(number_page=2):
	"""Input number of pages
	Outputs: list of 50*number_page links  """
	# Using requests module for downloading webpage content
	links=[]
	for page_number in range(1,number_page+1):
		url='https://stackoverflow.com/questions?sort=votes&pagesize=50&page={}'.format(page_number)
		response = requests.get(url)
    
    # Getting status of the request
    # 200 status code means our request was successful
    # 404 status code means that the resource you were looking for was not found
		response.status_code
    
    # Parsing html data using BeautifulSoup
		soup = bs(response.content, 'html.parser')
    
    # body
		body = soup.select_one('body')

		
		question_links = body.select("h3 a.question-hyperlink")
		for a in question_links:
				links.append('https://stackoverflow.com'+a['href'])
	
	return links
	
def create_data(number_page):
  """Input number of pages
  Outputs: Data Frame with title, tags, the question, code in the question, the answer and code in the answer  """
  links=get_links(number_page=number_page)
  df = pd.DataFrame(columns=['title','tags','question','code_question','answer','code_answer'])
  for link in links:
    df = df.append(Scrap_WebPage(link), ignore_index=True)

  #Drop Question with no answer or only code in question
  df=df.dropna(subset=['question','answer','code_answer'])
  ##Clean code columns  
  df['code_question']=df['code_question'].str[0]
  df['code_answer']=df['code_answer'].str[0] 
  df = df.replace('', 'no')
  df['code_question']=df['code_question'].fillna('no')
  df['code_answer']=df['code_answer'].fillna('no')
  #One Hot Encodding on Tags columns  
  s = df['tags']
  mlb = MultiLabelBinarizer()
  df_tags=pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=df.index)
  df=pd.concat([df,df_tags], axis=1)
  df=df.drop(columns=['tags'])
  return df
	
def decontracted(phrase):
  """Input : sentence 
  Output Sentence
  decontract a word 
  >>> decontracted("won't")
  "will not"
  """
  # specific
  phrase = re.sub(r"won't", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)

  # general
  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'s", " is", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'m", " am", phrase)
  phrase = re.sub(r"\n", "", phrase)
  return phrase
	
def preprocess(sentence):
  """ INPUT : Sentence
  OUTPUT : Sentence
  Pre processing step for a pandas dataset """
  sentence=str(sentence)
  sentence = sentence.lower()
  sentence = decontracted(sentence)
  sentence=sentence.replace('{html}',"") 
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', sentence)
  rem_url=re.sub(r'http\S+', '',cleantext)
  rem_num = re.sub('[0-9]+', '', rem_url)
  tokenizer = RegexpTokenizer(r'\w+')
  tokens = tokenizer.tokenize(rem_num)  
  filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
  lemma_words=[lemmatizer.lemmatize(w,pos='v') for w in filtered_words]
  lemma_words=[lemmatizer.lemmatize(w) for w in lemma_words]
  return " ".join(lemma_words)
	
	
def main():
 
  print('Loading data...\n')

  df=create_data(number_page=50)

  print('Cleaning data...')
  df['title']=df['title'].map(lambda s:preprocess(s))
  df['question']=df['question'].map(lambda s:preprocess(s))
  df['answer']=df['answer'].map(lambda s:preprocess(s))
  df['answer'] = df['answer'].replace('', 'no')
  df=df.fillna('no')
        
  print('Saving data...\n ')
  df.to_csv('clean_data.csv')
        
  print('Cleaned data saved to database!')

if __name__ == '__main__':
  main()