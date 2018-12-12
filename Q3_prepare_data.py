import sys
import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score 

omdb = pd.read_json('omdb-data.json.gz', orient='record', lines=True)

def expand(row):
	locations = row['omdb_genres'] if isinstance(row['omdb_genres'], list) else [row['omdb_genres']]
	s = pd.Series(row['omdb_plot'], index=list(set(locations)))
	return s

data = omdb.apply(expand, axis=1).stack()
data = data.to_frame().reset_index(level=1, drop=False)
data.columns = ['omdb_genres', 'omdb_plot']
data.reset_index(drop=True, inplace=True)

# data.to_csv('Q3_data.csv', index=False, header=None)

cv = TfidfVectorizer(min_df=1, stop_words='english')
X = data['omdb_plot']
y = data['omdb_genres']

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(y_test)

x_traincv = cv.fit_transform(X_train)
a = x_traincv.toarray()


model = MultinomialNB()
model.fit(x_traincv, y_train)


x_testcv = cv.transform(X_test)
predict = model.predict(x_testcv)

print(predict.shape)
actual = np.array(y_test)

print(accuracy_score(actual, predict))





