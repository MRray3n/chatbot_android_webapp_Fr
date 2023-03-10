# %%
# things we need for NLP

#############
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import *
import re
from chardet import *
#############
import nltk
from nltk.stem.lancaster import *
stemmer=nltk.stem.SnowballStemmer('french')
from flask import Flask, render_template, request, jsonify
# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
# %%
# restore all of our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)
# %%
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='french')
# %%
def clean_up_sentence(sentence):
    # Tokeniser la phrase
    doc = word_tokenize(sentence,language="french")
    print(doc)
    # Retourner le texte de chaque phrase
    stopWords = list(stopwords.words('french'))
    print(type(stopWords))
    t= ['?','.','!','"',',',"'", ']','[',"''", '``']
    stopWords.extend(t)
    clean_words = []
    for token in doc:
        if token not in stopWords:
            clean_words.append(token)
    
    print(clean_words)
    #doc = word_tokenize(str(clean_words),language="french")
    
    print("doc=")
    print(doc)
    sentence_words= [stemmer.stem(X.lower()) for X in doc]
    #del sentence_words[0]
    #del sentence_words[len(sentence_words)-1]
    #del sentence_words[0]
    #del sentence_words[len(sentence_words)-1]
    print(sentence_words,type(sentence_words))
    #st= ""
    #for i in sentence_words :
     #   st= st + " " + i          
    return sentence_words
    
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    print(np.array(bag))
    return(np.array(bag))

# %%
# load our saved model
model.load('./model.tflearn')

# %%
# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.40
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return quote(random.choice(i['responses']))
                    else :
                        return "je ne comprends pas ce que tu dis é"
    
            results.pop(0)
    else :
        return "je ne comprends pas ce que tu dis é"   
from flask_cors import CORS
#import unicodedata
from urllib.parse import *
app = Flask(__name__)
CORS(app)
app.static_folder = 'static'
@app.route("/") 
def home():
    return render_template("base.html")
@app.post("/")
def get_bot_response():
    userText = request.get_json().get("message")
    #userText = decoding(userText)
    #print(userText)
    #userText = userText.decode(encoding='UTF-8')
    ####################################### decode
     
    print(userText,type(userText))
    #decoded_text = bytes(userText,'utf-8').decode('utf-8')
    decoded_text = unquote(userText)
    print(decoded_text) 
    ''' 
    decoded_text = {}
    str_text= ""
    str(str_text) 
    print(userText)
    for key, value in userText.items():
        decoded_text[key] = chr(value)
        str_text = str_text + str(decoded_text[key])
    ''' 
    #######################################
    RES=response(str(decoded_text),userID='123', show_details=False)
    print (RES)
    print (jsonify(RES))
    mes = {"answer": RES}
    #str(mes.items).encode("utf-8")
    print(mes)
    return jsonify(mes)
@app.get("/get")
def responsechat():
        userText = request.args.get('msg')
        #print(userText)
        res = response(str(userText),userID='123', show_details=False)
        #print(res)
        #print(jsonify(cnt=res))
        return jsonify(cnt=res) 
if __name__ == "__main__":
    app.run(debug=True,port=7777)

  


#while True:
  #  ques = input("rayos-: ")
 #   print(classify(ques))
 #   response(ques)
