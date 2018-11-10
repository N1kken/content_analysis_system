import pandas as pd

from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    classifier.fit(feature_vector_train, label)
    joblib.dump(classifier, 'my_model.pkl', compress=9)
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, valid_y)


def classify(text):
    tests = []
    model_clone = joblib.load('my_model.pkl')
    tests.append(text.split())
    new = pd.DataFrame()
    new['test'] = tests
    xvalid_count2 = count_vect.transform(new['test'])
    classify = model_clone.predict(xvalid_count2)
    return encoder.inverse_transform(classify)


data = open('./corpus').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(content[1:])

trainDF = pd.DataFrame()
trainDF['text'] = texts
# print texts
trainDF['label'] = labels
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.15)

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', tokenizer=lambda doc: doc, lowercase=False)
count_vect.fit(trainDF['text'])
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

test = raw_input("Type your comment here: ")

# accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
# print "NB, Count Vectors: ", accuracy

print "This comment is: " + classify(test)