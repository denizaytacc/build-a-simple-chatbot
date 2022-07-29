import json # working with the json file
import pickle # saving python objects
import string # removing punctuations
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np # array structure
# NLP stuff
import nltk
from nltk.stem import WordNetLemmatizer

# Neural network building
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

vocabulary = list()
classes = list()
corpus = list()

for intent in intents["intents"]:
    classes.append(intent["tag"])
    for pattern in intent["patterns"]:
        pattern = pattern.lower()
        pattern = pattern.translate(str.maketrans('','',string.punctuation)) # to remove punctations
        tokenized_sentence = nltk.word_tokenize(pattern) # ex: I am better" -> ["I", "am", "better"]
        final_sentence = [lemmatizer.lemmatize(word) for word in tokenized_sentence] # ex: ["I", "am", "good"] -> ["I", "am", "good"]
        vocabulary.extend(final_sentence)
        corpus.append((final_sentence, intent["tag"]))

classes = sorted(classes)
vocabulary = sorted(set(vocabulary))
print("Classes:", classes)
print("Vocabulary:", vocabulary)

pickle.dump(classes, open("classes.pkl", "wb"))
pickle.dump(vocabulary, open("vocabulary.pkl", "wb"))

train_x = list()
train_y = list()

for sentence, sentence_class in corpus:
    class_idx = [0] * len(classes)
    bag_of_words = [0] * len(vocabulary)
    for word in sentence:
        if word in vocabulary:
            bag_of_words[vocabulary.index(word)] += 1
    class_idx[classes.index(sentence_class)] = 1
    train_x.append(bag_of_words)
    train_y.append(class_idx)

train_x = np.array(train_x)
train_y = np.array(train_y)

print(datetime.now().strftime("%H:%M:%S"), "Started building the model!")
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(len(train_y[0]), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

finished_model = model.fit(train_x, train_y, epochs=100, batch_size=6, verbose=0)
model.save("trained_model.h5", finished_model)
print(datetime.now().strftime("%H:%M:%S"), "Model is saved to trained_model.h5")