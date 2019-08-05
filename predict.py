from sys import argv
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import numpy as np

script, training_file, holdout_file = argv

def interact():
    """Open a REPL in the current context"""
    import code
    code.InteractiveConsole(locals=globals()).interact()

def open_csv(file_name):
    """
    Open a file, parse it as a comma delimeted structured array.
    Interoplate types and fill blank values with 0s
    """
    file_handle = open(file_name)
    out = np.genfromtxt(file_handle, dtype=None, filling_values="0", skip_header=1, delimiter=",", invalid_raise=False)
    file_handle.close()
    return out

def voidToTuple(x):
    """Cast numpy.void as tuple"""
    return tuple(i for i in x)

def build_dictionary(feature_columns, data, row_transform):
    """Convert a structured array to a dictionary for the vectorizer"""
    return [dict(zip(feature_columns, row_transform(x))) for x in data]

def build_features(vectorizer, data):
    """Load file, vectorize"""
    feature_columns = data.dtype.names[0:-1]
    feature_dicts = build_dictionary(feature_columns, data, lambda x: voidToTuple(x)[0:-1])
    return vectorizer.fit_transform(feature_dicts)

def build_model(features, data):
    """Build, crossvalidate and return the model pipeline"""
    targets = [x[-1] for x in data]

    stages = [
        ('standardize', StandardScaler(with_mean=False)),
        ('dimReduction', TruncatedSVD(n_components=100)),
        ('MLP', MLPClassifier(solver='adam', random_state=1))
    ]

    model_pipeline = Pipeline(stages)

    scores = cross_val_score(model_pipeline, features, targets, cv=5, scoring='roc_auc')
    print(scores)

    return model_pipeline.fit(features, targets)

def evaluate_holdout(model, vectorizer):
    """run the holdout data and save the result"""
    dat = open_csv(holdout_file)
    feature_columns = dat.dtype.names

    holdout_dicts = build_dictionary(feature_columns, dat, lambda x: voidToTuple(x))

    holdout = vectorizer.transform(holdout_dicts)
    holdout_prediction = model.predict_proba(holdout)

    output = "\n".join([str(number[1]) for number in holdout_prediction])
    text_file = open("out.txt", "w")
    text_file.write(output)
    text_file.close()


data = open_csv(training_file)
vectorizer = DictVectorizer()
features = build_features(vectorizer, data)
model = build_model(features, data)
evaluate_holdout(model, vectorizer)

#Open up the repl in this context so I can mess with things.
interact()
