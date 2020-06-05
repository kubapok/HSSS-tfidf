import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

DATA_PATH = '../petite-difference-challenge2/'


train_f = [a.rstrip('\n') for a in open(DATA_PATH + 'train/in.tsv', 'r').readlines()]
dev_f = [a.rstrip('\n') for a in open(DATA_PATH + 'dev-0/in.tsv', 'r').readlines()]
test_f = [a.rstrip('\n') for a in open(DATA_PATH + 'test-A/in.tsv', 'r').readlines()]


vectorizer = TfidfVectorizer()


train_X = vectorizer.fit_transform(train_f)
dev_X = vectorizer.transform(dev_f)
test_X = vectorizer.transform(test_f)


pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
pickle.dump(train_X,open('train_X.pkl','wb'))
pickle.dump(dev_X,open('dev_X.pkl','wb'))
pickle.dump(test_X,open('test_X.pkl','wb'))

train_y = np.array([int(a.rstrip('\n')) for a in open(DATA_PATH + 'train/expected.tsv').readlines()])
dev_y = np.array([int(a.rstrip('\n')) for a in open(DATA_PATH + 'dev-0/expected.tsv').readlines()])

model = LogisticRegression()
model.fit(train_X, train_y)

pickle.dump(model,open('model.pkl','wb'))

predicted_y = np.minimum(model.predict(dev_X), np.max(train_y))
predicted_y = np.maximum(predicted_y, np.min(train_y))

print('dev score:')
print(np.sqrt(mean_squared_error(predicted_y, dev_y)))
