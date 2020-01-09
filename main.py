
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# get data
data = pd.read_csv('.data/Data-WikipediaComments/train.csv')
COMMENT = 'comment_text'
data[COMMENT].fillna("unknown", inplace=True)
train, test = train_test_split(data, test_size=0.10, random_state=42)

# preprocess data
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

# vectorize data
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
tst_term_doc = vec.transform(test[COMMENT])

# save vertorize files
with open('.artifacts/vectorizer.pickle', 'wb') as handle:
    pickle.dump(vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

# run model
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc
x_t= tst_term_doc

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

label_cols=['toxic']

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    filename=".artifacts/"+j+".pickle"
    print(filename)
    pickle.dump(m, open(filename, 'wb'))

# publish results
from sklearn.metrics import accuracy_score
results=[accuracy_score(train["toxic"], m.predict(x)),accuracy_score(test["toxic"], m.predict(x_t))]
results=pd.DataFrame(results)
results=(results.T)
results.columns=["Train Result","Test Result"]
results.to_csv(".artifacts/result.csv",index=False)
