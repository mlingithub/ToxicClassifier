import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# get data
class_names = ['toxic']
data = pd.read_csv('.data/Data-WikipediaComments/train.csv').fillna(' ')
train, test = train_test_split(data, test_size=0.10, random_state=42)
train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

#vectorize data
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

# save vertorize files
with open('.artifacts/word_vectorizer.pickle', 'wb') as handle:
    pickle.dump(word_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('.artifacts/char_vectorizer.pickle', 'wb') as handle:
    pickle.dump(char_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# run model
scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]
    filename=".artifacts/"+class_name+".pickle"
    pickle.dump(classifier, open(filename, 'wb'))
    print("done")

print('Total CV score is {}'.format(np.mean(scores)))

# publish on wiki
from sklearn.metrics import accuracy_score
results=[accuracy_score(train["toxic"], classifier.predict(train_features)),accuracy_score(test["toxic"], classifier.predict(test_features))]
results=pd.DataFrame(results)
results=(results.T)
results.columns=["Train Result","Test Result"]
results.to_csv(".artifacts/result.csv",index=False)
