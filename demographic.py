import pandas as pd

df = pd.read_csv('article.csv')

df = df.sort_values(["total_events"], ascending = [False])

output = df[['uri', 'title', 'text', 'lang', 'total_events']].head(20).values.tolist()
