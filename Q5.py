import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix

#which genre gets more/less popular over time?
#which genre gets higher/lower reviews therefore the production is better/worse?
#most popular genre of film to make?

def to_timestamp_single(date):
	return date.timestamp()
to_timestamp = np.vectorize(to_timestamp_single)
'''
def delete_brackets(str):
	return str[1:-1]
'''
wiki = pd.read_json(sys.argv[1], orient = 'record', lines = True)
rotten_tomato = pd.read_json(sys.argv[2], orient = 'record', lines = True)
genre = pd.read_json(sys.argv[3], orient = 'record', lines = True)
rotten_tomato = rotten_tomato.set_index('imdb_id')

wiki_review = wiki.join(rotten_tomato, on ='imdb_id', lsuffix='_caller', rsuffix='_other')
wiki_review = wiki_review[wiki_review['rotten_tomatoes_id_caller'] == wiki_review['rotten_tomatoes_id_other']]
wiki_review['audience_average'] = wiki_review['audience_average']/5
wiki_review['audience_percent'] = wiki_review['audience_percent']/100
wiki_review['critic_average'] = wiki_review['critic_average']/10
wiki_review['critic_percent'] = wiki_review['critic_percent']/100


cols_to_keep = ['genre', 'made_profit', 'publication_date',
					'audience_average', 'audience_percent', 'audience_ratings','critic_average','critic_percent']
wiki_review = wiki_review[cols_to_keep]


wiki_review = wiki_review[pd.notnull(wiki_review['audience_average'])]
wiki_review = wiki_review[pd.notnull(wiki_review['audience_percent'])]
wiki_review = wiki_review[pd.notnull(wiki_review['audience_ratings'])]
wiki_review = wiki_review[pd.notnull(wiki_review['critic_average'])]
wiki_review = wiki_review[pd.notnull(wiki_review['critic_percent'])]
wiki_review = wiki_review[pd.notnull(wiki_review['publication_date'])]



wiki_review['publication_date'] = pd.to_datetime(wiki_review['publication_date'])
wiki_review['timestamp'] = wiki_review['publication_date'].apply(to_timestamp_single)


#DORIS: separating genres into 11 columns each with 1 or 0 genre for each movie
wiki_review = wiki_review.sort_values('genre')
wiki_review['genre_length'] = wiki_review.apply(lambda x: len(x['genre']), axis=1)
print(wiki_review['genre_length'].max())
print(wiki_review['genre_length'].min())


wiki_review[['genre1', 'genre2', 'genre3','genre4', 'genre5', 'genre6','genre7', 'genre8', 'genre9','genre10', 'genre11']] = pd.DataFrame(wiki_review['genre'].values.tolist(), index=wiki_review.index)
wiki_review = wiki_review.join(genre.set_index('wikidata_id'), on ='genre1')
wiki_review = wiki_review.join(genre.set_index('wikidata_id'), on ='genre2', rsuffix='2')
wiki_review = wiki_review.join(genre.set_index('wikidata_id'), on ='genre3', rsuffix='3')
wiki_review = wiki_review.join(genre.set_index('wikidata_id'), on ='genre4', rsuffix='4')
wiki_review = wiki_review.join(genre.set_index('wikidata_id'), on ='genre5', rsuffix='5')
wiki_review = wiki_review.join(genre.set_index('wikidata_id'), on ='genre6', rsuffix='6')
wiki_review = wiki_review.join(genre.set_index('wikidata_id'), on ='genre7', rsuffix='7')
wiki_review = wiki_review.join(genre.set_index('wikidata_id'), on ='genre8', rsuffix='8')
wiki_review = wiki_review.join(genre.set_index('wikidata_id'), on ='genre9', rsuffix='9')
wiki_review = wiki_review.join(genre.set_index('wikidata_id'), on ='genre10', rsuffix='10')
wiki_review = wiki_review.join(genre.set_index('wikidata_id'), on ='genre11', rsuffix='11')

wiki_review = wiki_review.drop(['genre', 'genre_length', 'genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8', 'genre9', 'genre10', 'genre11'], axis=1)


genres_count = pd.concat([wiki_review['genre_label'], wiki_review['genre_label2'], wiki_review['genre_label3'],wiki_review['genre_label4'], wiki_review['genre_label5'], 
wiki_review['genre_label6'], wiki_review['genre_label7'], wiki_review['genre_label8'],wiki_review['genre_label9'], wiki_review['genre_label10'], wiki_review['genre_label11'],])
#DORIS: ends here



#which type of film gets made the most?
print(genres_count.value_counts())
'''
These genres are of interests because they are the most popular types to be produced in the industry.
drama film                    6272
comedy film                   2968
action film                   1762
horror film                   1660
documentary film              1340
crime film                    1334
thriller film                 1304
film based on literature      1263
comedy-drama                  1241
romantic comedy               1185
science fiction film          1171
LGBT-related film             1088
'''

#now we extract thse particular 12 types of movie and store them in 12 dataframes

drama = wiki_review[(wiki_review['genre_label']=='drama film')|(wiki_review['genre_label2']=='drama film')|(wiki_review['genre_label3']=='drama film')|(wiki_review['genre_label4']=='drama film')|
					(wiki_review['genre_label5']=='drama film')|(wiki_review['genre_label6']=='drama film')|(wiki_review['genre_label7']=='drama film')|(wiki_review['genre_label8']=='drama film')|
					(wiki_review['genre_label9']=='drama film')|(wiki_review['genre_label10']=='drama film')|(wiki_review['genre_label11']=='drama film')]


comedy = wiki_review[(wiki_review['genre_label']=='comedy film')|(wiki_review['genre_label2']=='comedy film')|(wiki_review['genre_label3']=='comedy film')|(wiki_review['genre_label4']=='comedy film')|
					(wiki_review['genre_label5']=='comedy film')|(wiki_review['genre_label6']=='comedy film')|(wiki_review['genre_label7']=='comedy film')|(wiki_review['genre_label8']=='comedy film')|
					(wiki_review['genre_label9']=='comedy film')|(wiki_review['genre_label10']=='comedy film')|(wiki_review['genre_label11']=='comedy film')]

action = wiki_review[(wiki_review['genre_label']=='action film')|(wiki_review['genre_label2']=='action film')|(wiki_review['genre_label3']=='action film')|(wiki_review['genre_label4']=='action film')|
					(wiki_review['genre_label5']=='action film')|(wiki_review['genre_label6']=='action film')|(wiki_review['genre_label7']=='action film')|(wiki_review['genre_label8']=='action film')|
					(wiki_review['genre_label9']=='action film')|(wiki_review['genre_label10']=='action film')|(wiki_review['genre_label11']=='action film')]

horror = wiki_review[(wiki_review['genre_label']=='horror film')|(wiki_review['genre_label2']=='horror film')|(wiki_review['genre_label3']=='horror film')|(wiki_review['genre_label4']=='horror film')|
					(wiki_review['genre_label5']=='horror film')|(wiki_review['genre_label6']=='horror film')|(wiki_review['genre_label7']=='horror film')|(wiki_review['genre_label8']=='horror film')|
					(wiki_review['genre_label9']=='horror film')|(wiki_review['genre_label10']=='horror film')|(wiki_review['genre_label11']=='horror film')]

documentary = wiki_review[(wiki_review['genre_label']=='documentary film')|(wiki_review['genre_label2']=='documentary film')|(wiki_review['genre_label3']=='documentary film')|(wiki_review['genre_label4']=='documentary film')|
					(wiki_review['genre_label5']=='documentary film')|(wiki_review['genre_label6']=='documentary film')|(wiki_review['genre_label7']=='documentary film')|(wiki_review['genre_label8']=='documentary film')|
					(wiki_review['genre_label9']=='documentary film')|(wiki_review['genre_label10']=='documentary film')|(wiki_review['genre_label11']=='documentary film')]

crime = wiki_review[(wiki_review['genre_label']=='crime film')|(wiki_review['genre_label2']=='crime film')|(wiki_review['genre_label3']=='crime film')|(wiki_review['genre_label4']=='crime film')|
					(wiki_review['genre_label5']=='crime film')|(wiki_review['genre_label6']=='crime film')|(wiki_review['genre_label7']=='crime film')|(wiki_review['genre_label8']=='crime film')|
					(wiki_review['genre_label9']=='crime film')|(wiki_review['genre_label10']=='crime film')|(wiki_review['genre_label11']=='crime film')]

thriller = wiki_review[(wiki_review['genre_label']=='thriller film')|(wiki_review['genre_label2']=='thriller film')|(wiki_review['genre_label3']=='thriller film')|(wiki_review['genre_label4']=='thriller film')|
					(wiki_review['genre_label5']=='thriller film')|(wiki_review['genre_label6']=='thriller film')|(wiki_review['genre_label7']=='thriller film')|(wiki_review['genre_label8']=='thriller film')|
					(wiki_review['genre_label9']=='thriller film')|(wiki_review['genre_label10']=='thriller film')|(wiki_review['genre_label11']=='thriller film')]

romcom = wiki_review[(wiki_review['genre_label']=='romantic comedy')|(wiki_review['genre_label2']=='romantic comedy')|(wiki_review['genre_label3']=='romantic comedy')|(wiki_review['genre_label4']=='romantic comedy')|
					(wiki_review['genre_label5']=='romantic comedy')|(wiki_review['genre_label6']=='romantic comedy')|(wiki_review['genre_label7']=='romantic comedy')|(wiki_review['genre_label8']=='romantic comedy')|
					(wiki_review['genre_label9']=='romantic comedy')|(wiki_review['genre_label10']=='romantic comedy')|(wiki_review['genre_label11']=='romantic comedy')]

scifi = wiki_review[(wiki_review['genre_label']=='science fiction film')|(wiki_review['genre_label2']=='science fiction film')|(wiki_review['genre_label3']=='science fiction film')|(wiki_review['genre_label4']=='science fiction film')|
					(wiki_review['genre_label5']=='science fiction film')|(wiki_review['genre_label6']=='science fiction film')|(wiki_review['genre_label7']=='science fiction film')|(wiki_review['genre_label8']=='science fiction film')|
					(wiki_review['genre_label9']=='science fiction film')|(wiki_review['genre_label10']=='science fiction film')|(wiki_review['genre_label11']=='science fiction film')]

LGBT = wiki_review[(wiki_review['genre_label']=='LGBT-related film')|(wiki_review['genre_label2']=='LGBT-related film')|(wiki_review['genre_label3']=='LGBT-related film')|(wiki_review['genre_label4']=='LGBT-related film')|
					(wiki_review['genre_label5']=='LGBT-related film')|(wiki_review['genre_label6']=='LGBT-related film')|(wiki_review['genre_label7']=='LGBT-related film')|(wiki_review['genre_label8']=='LGBT-related film')|
					(wiki_review['genre_label9']=='LGBT-related film')|(wiki_review['genre_label10']=='LGBT-related film')|(wiki_review['genre_label11']=='LGBT-related film')]



DataFrames = drama,comedy,action,horror,documentary,crime,thriller,romcom,scifi,LGBT
Names = ['drama', 'comedy', 'action', 'horror', 'documentary', 'crime', 'thriller', 'romcom', 'scifi', 'LGBT']
count = 0

for df in DataFrames:
	print(Names[count])
	df = df.sort_values('timestamp')
	df = df[['publication_date','timestamp','audience_ratings','audience_percent','audience_average','critic_percent','critic_average']]
	df['year'] = df['publication_date'].dt.year
	df = df.groupby('year').mean()
	df = df.reset_index()
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
	ax1.plot(df['year'], df['critic_average'])
	ax2.plot(df['year'], df['critic_percent'])
	ax3.plot(df['year'], df['audience_average'])
	ax4.plot(df['year'], df['audience_percent'])
	#data are more concentrated in later (more recent) time periods since more films are made
	plt.show()
	plt.plot(df['year'], df['audience_ratings'])
	plt.show()
	print('audience average ratings linear regression slope:', stats.linregress(df['timestamp'],df['audience_average']).slope)
	print('audience average ratings linear regression p-value:', stats.linregress(df['timestamp'],df['audience_average']).pvalue)
	print('audience percentage linear regression slope:', stats.linregress(df['timestamp'],df['audience_percent']).slope)
	print('audience percentage linear regression p-value:', stats.linregress(df['timestamp'],df['audience_percent']).pvalue)
	print('critic average ratings linear regression slope:', stats.linregress(df['timestamp'],df['critic_average']).slope)
	print('critic average ratings linear regression p-value:', stats.linregress(df['timestamp'],df['critic_average']).pvalue)
	print('critic percentage linear regression slope:', stats.linregress(df['timestamp'],df['critic_percent']).slope)
	print('critic percentage linear regression p-value:', stats.linregress(df['timestamp'],df['critic_percent']).pvalue)
	print('count of audience reviews slope:', stats.linregress(df['timestamp'],df['critic_percent']).slope)
	print('count of audience reviews p-value:', stats.linregress(df['timestamp'],df['critic_percent']).pvalue)
	count = count+1


