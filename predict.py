import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import LancasterStemmer
import numpy as np
import pickle
import random
import json
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import time
import os

with open("intents.json") as f:
    data = json.load(f)

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

stemmer = LancasterStemmer()
model = load_model("model.h5")

r = sr.Recognizer()

def bag_of_words(s, words):
    bag = np.zeros(len(words))
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array([bag])


def chat():
    print("Start talking with the bot (say quit to stop)")
    while True:
        print("You: ")
        with sr.Microphone() as mic:
            r.adjust_for_ambient_noise(mic, duration=0.2)
            audio = r.listen(mic, phrase_time_limit=4)
            inp = r.recognize_google(audio)
            inp = inp.lower()
        if inp == "quit":
            tts = gTTS(text="Bye!", lang='en')
            tts.save("text.mp3")
            playsound("text.mp3")
            os.remove("text.mp3")
            break
        else:
            results = model.predict([bag_of_words(inp, words)])
            results_index = np.argmax(results)
            tag = labels[results_index]
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            tts = gTTS(text=random.choice(responses), lang='en')
            tts.save("text.mp3")
            playsound("text.mp3")
            os.remove("text.mp3")
            time.sleep(2)
            
chat()
