import json
import pickle
import string 
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

vocabulary = pickle.load(open("vocabulary.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

model = load_model("trained_model.h5")


def print_chatbot_answer(intention):
    for intent in intents["intents"]:
        if intent["tag"] == intention:
            print("Bot >", random.choice(intent["responses"]))
            return 1 if intent["tag"] == "goodbye" else 0

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('','',string.punctuation)) # to remove punctations
    tokenized_sentence = nltk.word_tokenize(sentence)
    final_sentence = [lemmatizer.lemmatize(word) for word in tokenized_sentence]
    return final_sentence

def transform_to_bag_of_words(sentence):
    sentence = clean_sentence(sentence)
    bag = [0] * len(vocabulary)
    for word in sentence:
        if word in vocabulary:
            bag[vocabulary.index(word)] += 1
    return np.array(bag)

def predict(sentence):
    bag_of_words = transform_to_bag_of_words(sentence)
    result = model.predict(np.array([bag_of_words]), verbose=0)[0]
    acceptable_rate = 0.2
    return result.argmax() if result[result.argmax()] > 0.2 else -1

if __name__ == "__main__":
    user_quit = False
    print("Enter your input.")
    while user_quit == False:
        user_input = input("You > ")
        class_idx = predict(user_input)
        if class_idx == -1:
            print("I couldn't understand you, could you repeat in a different manner?")
        else:
            user_quit = print_chatbot_answer(classes[class_idx])