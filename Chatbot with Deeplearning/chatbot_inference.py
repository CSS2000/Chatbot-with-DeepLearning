import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle
import colorama
from colorama import Fore, Style
import random

# Initialize colorama
colorama.init()

# Load data
with open('intents.json') as file:
    data = json.load(file)

# Load model and supporting files
model = keras.models.load_model('chat_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc_file:
    lbl_encoder = pickle.load(enc_file)

# Chat function
def chat():
    max_len = 20
    print(Fore.YELLOW + "Start messaging with the bot (type 'quit' to stop)!" + Style.RESET_ALL)

    while True:
        user_input = input(Fore.LIGHTBLUE_EX + "You: " + Style.RESET_ALL)
        if user_input.lower() == 'quit':
            break

        # Predict the intent
        padded_input = keras.preprocessing.sequence.pad_sequences(
            tokenizer.texts_to_sequences([user_input]),
            truncating='post', maxlen=max_len
        )
        result = model.predict(padded_input, verbose=0)  # Suppress prediction logs
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        # Respond based on intent
        for intent in data['intents']:
            if intent['tag'] == tag[0]:
                print(Fore.GREEN + "ChatBot: " + Style.RESET_ALL, random.choice(intent['responses']))

# Start chat
if __name__ == "__main__":
    chat()
