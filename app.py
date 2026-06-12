import streamlit as st
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_LEN = 100

# Load Model
model = load_model("spam_lstm.keras")

# Load Tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Streamlit UI
st.title("📩 Spam Detection using LSTM")

st.write(
    "Enter a message below and check whether it is Spam or Not spam."
)

message = st.text_area("Enter SMS Message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        sequence = tokenizer.texts_to_sequences([message])

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding='post'
        )

        prediction = model.predict(padded)

        score = prediction[0][0]

        if score > 0.5:
            st.error(
                f"Spam Message\nConfidence: {score:.2%}"
            )
        else:
            st.success(
                f"Not spam Message\nConfidence: {(1-score):.2%}"
            )