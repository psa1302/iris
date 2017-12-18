# Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():

	"""Load the Iris dataset and assign proper variables after splitting it
	into test data and validation data with 80% of data being used to test and 
	20% data used for validation."""

	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = pd.read_csv(url, names=names).replace({'class': {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}})
	training_data, validation_data = train_test_split(dataset, test_size=0.2)
	return (training_data, validation_data, dataset)

def vectorized_result(j):

	"""To vectorize the ouput."""

	z = np.zeros((3, 1))
	z[int(j)] = 1.0
	return z

def load_data_wrapper():

	"""Data modification for neural network. Converting the input into a 
	column vector(4,1) and the output into a column vector(3,1)."""
	
	tr_d, va_d, te_d = load_data()
	training_d = np.split(tr_d, [4], axis=1)
	validation_d = np.split(va_d, [4], axis=1)
	training_inputs = [np.reshape(x, (4,1)) for x in training_d[0].values]
	training_results = [vectorized_result(y) for y in training_d[1].values]
	training_data = list(zip(training_inputs, training_results))
	validation_inputs = [np.reshape(x, (4, 1)) for x in validation_d[0].values]
	validation_results = [vectorized_result(y) for y in validation_d[1].values]
	validation_data = list(zip(validation_inputs, validation_results))
	return (training_data, validation_data)