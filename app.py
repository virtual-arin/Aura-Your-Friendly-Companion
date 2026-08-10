import streamlit as st
import json
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Load Resources (Cached) ---
# We cache this so the heavy Keras model doesn't reload on every chat interaction
@st.cache_resource
def load_bot_resources():
    with open('data/intents.json') as file:
        data = json.load(file)
   
    model = keras.models.load_model('models/chat_model.h5')

    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('models/label_encoder.pickle', 'rb') as enc:
        label_encoder = pickle.load(enc)
        
    return data, model, tokenizer, label_encoder

data, model, tokenizer, label_encoder = load_bot_resources()

# --- 2. Build the Streamlit App ---
st.title("✨ Meet Aura")
st.caption("🚀 Your intuitive AI companion for data, coding, and creativity.")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("Enter your message:"):
    
    # 1. Display user message in chat UI
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # 2. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 3. Process the input using your ML model
    padded_sequences = pad_sequences(
        tokenizer.texts_to_sequences([user_input]), 
        truncating='post', 
        maxlen=20
    )
    
    # Predict the intent
    prediction = model.predict(padded_sequences, verbose=0)
    # result is returned as an array, we need the first element
    result = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    # Find the appropriate response from intents.json
    bot_response = "I'm not sure how to respond to that." # Fallback response
    for intent in data['intents']:
        if intent['tag'] == result:
            bot_response = random.choice(intent['responses'])
            break
            
    # 4. Display the bot's response in chat UI
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        
    # 5. Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})