import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import accuracy_score
# from pyspark.sql import SparkSession,functions,types
# spark = SparkSession.builder.appName('movies').getOrCreate()
# assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
# assert spark.version >= '2.1' # make sure we have Spark 2.1+
# rotten_tomato = spark.read.json('rotten-tomatoes.json.gz')
# rotten_tomato.select("audience_average").show()
# print('\n')
# rotten_tomato.printSchema()
# rotten_tomato.describe().show()

filename1 = sys.argv[1]

# Read input file and fix some data points
data = pd.read_json(filename1, orient = 'record', lines = True)
data = data.dropna(how = 'any')
data['audience_average'] = data['audience_average']/5
data['audience_percent'] = data['audience_percent']/100
data['critic_average'] = data['critic_average']/10
data['critic_percent'] = data['critic_percent']/100

data = data[pd.notnull(data['audience_average'])]
data = data[pd.notnull(data['audience_percent'])]
data = data[pd.notnull(data['critic_average'])]
data = data[pd.notnull(data['critic_percent'])]

criteria = data[['audience_average','audience_percent','critic_average','critic_percent']]
print('criteria:')
print(criteria)
print('criteria.describe()')
print(criteria.describe())

#Basically plot a scatter plot and see that if there has a positive cluster or negative cluster between x and y we choose from our data set.
#plt.scatter(data['audience_average'], data['critic_average'])
#plt.scatter(data['audience_percent'], data['critic_percent'])
#plt.show()

#test pairwise correlation among the criteria
corr = criteria.corr()
print('Correlation coefficient between each pair:')
print(corr)


#This is to produce a matix of scatterplots of the correlation between each pair of variables
data_plot = criteria[['audience_average','audience_percent','critic_average','critic_percent']]
scatter_matrix(data_plot, alpha = 0.2, figsize = (15, 15), diagonal = 'kde')
plt.savefig("scatter_matrix.png")


#polynomial fit on audience_average and audience_percent
N = 3
x = criteria['audience_average']
X = x[:, np.newaxis]
y = criteria['audience_percent']
X_train, X_test, y_train, y_test = train_test_split(X, y)

model_poly = make_pipeline(
    PolynomialFeatures(degree=N, include_bias=True),
    LinearRegression(fit_intercept=False)
	)
model_poly.fit(X_train, y_train)
print('accuracy score for N=3 for audience_average and audience_percent:',model_poly.score(X_test,y_test))

#polynomial fit on critic_average and critic_percent
N = 3
x = criteria['critic_average']
X = x[:, np.newaxis]
y = criteria['critic_percent']
X_train, X_test, y_train, y_test = train_test_split(X, y)
model_poly.fit(X_train, y_train)
print('accuracy score for N=3 for audience_average and audience_percent:',model_poly.score(X_test,y_test))

#test linear regression between critic_average and audience_average
print("Test LinearRegression between audience_average and critic_average:")
reg = stats.linregress(criteria['audience_average'], criteria['critic_average'])
print("Print regression line slope")
print(reg.slope)
print("print regression line intercept:")
print(reg.intercept)
print("print regression r-value:")
print(reg.rvalue)
print("print regression p-value:")
print(reg.pvalue)

#using machine learning to test for accuracy score
X = criteria['audience_average'][:, np.newaxis]
y = criteria['critic_average']
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
model_lin = LinearRegression(fit_intercept=True)
model_lin.fit(X_train, y_train)
print('Using machine learning linear regression to confirm above slope and intercept')
print(model_lin.coef_[0], model_lin.intercept_)
print('accuracy score:', model_lin.score(X_test, y_test))

