import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = '../petite-difference-challenge2/'


train_f = [a.rstrip('\n') for a in open(DATA_PATH + 'train/in.tsv', 'r',newline = '\n').readlines()]
dev_f = [a.rstrip('\n') for a in open(DATA_PATH + 'dev-0/in.tsv', 'r',newline = '\n').readlines()]
dev1_f = [a.rstrip('\n') for a in open(DATA_PATH + 'dev-1/in.tsv', 'r',newline = '\n').readlines()]
test_f = [a.rstrip('\n') for a in open(DATA_PATH + 'test-A/in.tsv', 'r',newline = '\n').readlines()]

vectorizer = TfidfVectorizer()

train_X = vectorizer.fit_transform(train_f)
dev_X = vectorizer.transform(dev_f)
dev1_X = vectorizer.transform(dev1_f)
test_X = vectorizer.transform(test_f)

pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
pickle.dump(train_X,open('train_X.pkl','wb'))
pickle.dump(dev_X,open('dev0_X.pkl','wb'))
pickle.dump(dev1_X,open('dev1_X.pkl','wb'))
pickle.dump(test_X,open('test_X.pkl','wb'))

# END OF PREPROCESS

# TRAIN MODEL

train_X = pickle.load(open('train_X.pkl','rb'))
dev0_X = pickle.load(open('dev0_X.pkl','rb'))
dev1_X = pickle.load(open('dev1_X.pkl','rb'))
test_X = pickle.load(open('test_X.pkl','rb'))

train_y = np.array([int(a.rstrip('\n')) for a in open(DATA_PATH + 'train/expected.tsv',newline = '\n').readlines()])
dev_y = np.array([int(a.rstrip('\n')) for a in open(DATA_PATH + 'dev-0/expected.tsv',newline = '\n').readlines()])

model = LogisticRegression(max_iter = 100000,verbose=1)
model.fit(train_X, train_y)

pickle.dump(model,open('model.pkl','wb'))

### EVALUATE
predicted_y = model.predict(dev0_X)

print('dev score:')
print(accuracy_score(predicted_y, dev_y))
predicted_y = model.predict_proba(dev0_X)[:,1]
#import pdb; pdb.set_trace()
### OUTPUT
def predict(X,out_file_path):
    f_out  = open(out_file_path,'w')

    for p in model.predict_proba(X)[:,1]:
        f_out.write(str(p) + '\n')
    f_out.close()

predict(dev0_X,'dev-0/out.tsv')
predict(dev1_X,'dev-1/out.tsv')
predict(test_X,'test-A/out.tsv')

