
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import numpy as np

st.title("3-Player Rock Paper Scissors Predictor")

st.write("Input the sequence of winners by color (Red, Green, Blue), and I'll predict the next winner!")

# User input
user_input = st.text_input("Enter winners, separated by commas:", "Red, Green, Blue, Blue, Blue, Red, Green, Green, Green, Red")

if user_input:
    # Parse and encode the input
    try:
        winners = [w.strip().capitalize() for w in user_input.split(',')]
        le = LabelEncoder()
        encoded_winners = le.fit_transform(winners)

        # Check if we have enough data
        if len(encoded_winners) < 4:
            st.warning("Please enter at least 4 winners to make a prediction.")
        else:
            # Create training data
            X = []
            y = []
            for i in range(3, len(encoded_winners)):
                X.append(encoded_winners[i-3:i])  # 3-round history
                y.append(encoded_winners[i])      # next winner

            X = np.array(X)
            y = np.array(y)

            # Train model
            model = DecisionTreeClassifier()
            model.fit(X, y)

            # Predict
            last_3 = encoded_winners[-3:]
            prediction = model.predict([last_3])[0]
            predicted_color = le.inverse_transform([prediction])[0]

            st.success(f"Predicted next winner: {predicted_color}")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
