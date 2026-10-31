# -*- coding: utf-8 -*-
"""hearth_care_bots.py
Healthcare Symptom Triage Chatbot
"""

import os
import re
import random
import json
import pickle
import time
from pathlib import Path
import numpy as np

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    lemmatizer = None

def tokenize_words(text):
    if NLTK_AVAILABLE:
        try:
            return nltk.word_tokenize(text)
        except Exception:
            pass
    return re.findall(r'\b\w+\b', text)

def lemmatize_word(word):
    if NLTK_AVAILABLE and lemmatizer:
        try:
            return lemmatizer.lemmatize(word.lower())
        except Exception:
            pass
    return word.lower()

# Optional Speech and Audio imports
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Resolve paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
intents_path = os.path.join(MODULE_DIR, "intents.json")
if not os.path.exists(intents_path):
    intents_path = "intents.json"

with open(intents_path, "r", encoding="utf-8") as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = tokenize_words(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatize_word(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

def clean_up_sentence(sentence):
    sentence_words = tokenize_words(sentence)
    sentence_words = [lemmatize_word(word) for word in sentence_words]
    return sentence_words

def bow(sentence, words_vocab=None):
    if words_vocab is None:
        words_vocab = words
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words_vocab)
    for s in sentence_words:
        for i, w in enumerate(words_vocab):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    sentence_words = clean_up_sentence(sentence)
    best_intent = None
    best_score = -1

    for doc in documents:
        doc_words = [lemmatize_word(w) for w in doc[0]]
        overlap = sum(1 for w in sentence_words if w in doc_words)
        if overlap > best_score:
            best_score = overlap
            best_intent = doc[1]

    if best_intent and best_score > 0:
        prob = min(1.0, best_score / max(1, len(sentence_words)))
        return [{"intent": best_intent, "probability": f"{prob:.2f}"}]
    return [{"intent": "General Consultation", "probability": "0.50"}]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Please consult a healthcare professional for specific clinical advice."

def triage_symptom(text):
    """Programmatic non-interactive entrypoint for symptom triage."""
    preds = predict_class(text)
    response = get_response(preds, intents)
    return {
        "symptom_query": text,
        "predicted_intent": preds[0]["intent"],
        "confidence": float(preds[0]["probability"]),
        "triage_recommendation": response
    }

def calling_the_bot(text, engine=None):
    res_info = triage_symptom(text)
    print("Your Symptom was  : ", text)
    print("Result found in our Database : ", res_info["predicted_intent"], "->", res_info["triage_recommendation"])
    if engine and TTS_AVAILABLE:
        try:
            engine.say(f"Found it. From our database: {res_info['triage_recommendation']}")
            engine.runAndWait()
        except Exception:
            pass
    return res_info

if __name__ == '__main__':
    print("Bot is Running")
    sample_query = "fever, cough, headache"
    result = calling_the_bot(sample_query)
    print(f"Sample Triage Result: {result}")
