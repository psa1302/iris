import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = pd.read_csv(url, names=names)
	training_data, validation_data = train_test_split(dataset, test_size=0.2)
	test_data = dataset
	return (training_data, validation_data, dataset)

def array_conv(a, col1, col2):
	arr = [pd.DataFrame(a, columns=col1).values, pd.DataFrame(a, columns=col2).values]
	return arr

def load_data_wrapper():
	tr_d, va_d, te_d = load_data()
	class_data = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
	train_d = array_conv(tr_d, class_data, ['class'])
	valid_d = array_conv(va_d, class_data, ['class'])
	test_d = array_conv(te_d, class_data, ['class'])
	training_inputs = [np.reshape(x, (4, 1)) for x in train_d[0]]
	training_results = [vectorized_result(y) for y in train_d[1]]
	training_data = zip(training_inputs, training_results)
	validation_inputs = [np.reshape(x, (4, 1)) for x in valid_d[0]]
	validation_data = zip(validation_inputs, valid_d[1])
	test_inputs = [np.reshape(x, (4, 1)) for x in test_d[0]]
	test_data = zip(test_inputs, test_d[1])
	return (training_data, validation_data, test_data)

def vectorized_result(j):
	e = np.zeros((10, 1))
	if j == 'Iris-sentosa':
		e[0] = 1.0
	elif j == 'Iris-versicolor':
		e[1] = 1.0
	else:
		e[2] = 1.0
	return e

load_data_wrapper()