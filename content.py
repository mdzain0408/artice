import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

df = pd.read_csv('article.csv')
df = df[df['title'].notna()]
df.drop(['level_0'], axis=1, inplace=True)

count = CountVectorizer(stop_words = 'english')
countmatrix = count.fit_transform(df['title']) 

cosine_sim = cosine_similarity(countmatrix, countmatrix) 

df = df.reset_index()
indexes = pd.Series(df.index, index = df['contentId'])

def getrecomandations(contentId, cosine_sim):
  idx = indexes[contentId]
  SC = list(enumerate(cosine_sim[idx]))
  SC = sorted(SC, key = lambda x: x[1], reverse = True)
  SC = SC[1:11]
  Mi = [i[0] for i in SC ]
  return df[['uri', 'title', 'text', 'lang', 'total_events']].iloc[Mi].values.tolist()