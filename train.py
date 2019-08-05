from sys import argv
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import perceptron
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

def interact():
    import code
    code.InteractiveConsole(locals=globals()).interact()

script, training_file, holdout_file = argv

def open_csv(file_name):
	file_handle = open(file_name)
	data = np.genfromtxt(file_handle, dtype=None, filling_values="0", skip_header=1, delimiter=",", invalid_raise=False)
	file_handle.close()
	return data

def voidToTuple(v):
	return tuple(i for i in x)

data = open_csv(training_file)
holdout_data = open_csv(holdout_file)


feature_columns = data.dtype.names[0:-1]
target_column = data.dtype.names[-1]

#why do you numpy.voids have to look but not act like tuples..


feature_dicts = [dict(zip(feature_columns, voidToTuple(x)[0:-1])) for x in data]
holdout_dicts = [dict(zip(feature_columns, voidToTuple(x))) for x in holdout_data]

#instantiate your vectorizer, your targets, and your features.
vectorizer = DictVectorizer()
targets = [x[-1] for x in data]
features = vectorizer.fit_transform(feature_dicts)
holdout = vectorizer.transform(holdout_dicts)

print("building cross validation scores.")

estimators = [
	('standardize', StandardScaler(with_mean=False)),
	('dimReduction', TruncatedSVD(n_components=100)),
	('MLP', MLPClassifier(solver='adam', random_state=1))
]

model = Pipeline(estimators)
#scores = cross_val_score(model, features, targets, cv=5, scoring='roc_auc')
print(features)
model.fit(features,targets)

#pred = model.predict(features)
#CM = metrics.confusion_matrix(targets, pred)

#TN = CM[0][0]
#FN = CM[1][0]
#TP = CM[1][1]
#FP = CM[0][1]

holdout_prediction = model.predict(holdout)
output = "\n".join(["%d" % number for number in holdout_prediction])
text_file = open("output.txt", "w")
text_file.write(output)
text_file.close()


interact()
