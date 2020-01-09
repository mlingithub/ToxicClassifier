from flask import Flask, render_template, request, jsonify, flash, url_for, request, redirect,session
import numpy as np
import pickle as p
import json
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from scipy.sparse import hstack

# adaw

app = Flask(__name__)


modelfile = 'toxic.pickle'
model = p.load(open(modelfile, 'rb'))
#model._make_predict_function()
with open('word_vectorizer.pickle', 'rb') as handle:
    word_vectorizer = p.load(handle)
with open('char_vectorizer.pickle', 'rb') as handle:
    char_vectorizer = p.load(handle)

# maxlen = 100


@app.route("/", methods=['post','get'])
def makecalc():
    message=""
    color="black"

    if request.method == 'POST':

        stimulus = [request.form.get('textInput')]  # access the data inside

        word_features = word_vectorizer.transform(stimulus)
        char_features = char_vectorizer.transform(stimulus)
        features = hstack([char_features, word_features])
        #list_tokenized_text = tokenizer.texts_to_sequences([age])
        #X_text = pad_sequences(list_tokenized_text, maxlen=maxlen)
        
        prediction = model.predict(features)
        if prediction[0]==1:
            message= "Toxic"
            color="red"
        else:
            message= "Not Toxic"

    return render_template('predict.html', message=message, color=color)

@app.route('/api/', methods=['post'])
def makecalc2():
    data = request.get_json()
    prediction = np.array2string(model.predict(data))

    return jsonify(prediction)

if __name__ == '__main__':
    app.run(threaded=False)
