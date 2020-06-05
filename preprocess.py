import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

DATA_PATH = '../petite-difference-challenge2/'


train_f = [a.rstrip('\n') for a in open(DATA_PATH + 'train/in.tsv', 'r',newline = '\n').readlines()]
dev_f = [a.rstrip('\n') for a in open(DATA_PATH + 'dev-0/in.tsv', 'r',newline = '\n').readlines()]
test_f = [a.rstrip('\n') for a in open(DATA_PATH + 'test-A/in.tsv', 'r',newline = '\n').readlines()]

vectorizer = TfidfVectorizer()

train_X = vectorizer.fit_transform(train_f)
dev_X = vectorizer.transform(dev_f)
test_X = vectorizer.transform(test_f)

pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
pickle.dump(train_X,open('train_X.pkl','wb'))
pickle.dump(dev_X,open('dev_X.pkl','wb'))
pickle.dump(test_X,open('test_X.pkl','wb'))

# END OF PREPROCESS

# TRAIN MODEL

train_X = pickle.load(open('train_X.pkl','rb'))
dev_X = pickle.load(open('dev_X.pkl','rb'))
test_X = pickle.load(open('test_X.pkl','rb'))

train_y = np.array([int(a.rstrip('\n')) for a in open(DATA_PATH + 'train/expected.tsv',newline = '\n').readlines()])
dev_y = np.array([int(a.rstrip('\n')) for a in open(DATA_PATH + 'dev-0/expected.tsv',newline = '\n').readlines()])

model = LogisticRegression(max_iter = 100000,verbose=1)
model.fit(train_X, train_y)

pickle.dump(model,open('model.pkl','wb'))

### EVALUATE
predicted_y = np.minimum(model.predict(dev_X), np.max(train_y))
predicted_y = np.maximum(predicted_y, np.min(train_y))

print('dev score:')
print(np.sqrt(mean_squared_error(predicted_y, dev_y)))
### OUTPUT
def predict(X,out_file_path):
    f_out  = open(out_file_path,'w')
    predicted_y = np.minimum(model.predict(X), np.max(train_y))
    predicted_y = np.maximum(predicted_y, np.min(train_y))

    for p in predicted_y:
        f_out.write(str(p) + '\n')
    f_out.close()

predict(dev_X,'dev-0/out.tsv')
predict(test_X,'test-A/out.tsv')

