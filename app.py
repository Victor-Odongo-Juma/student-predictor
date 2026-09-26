import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# --- TRAIN MODEL ---
data = {
    'hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'marks': [20, 25, 30, 40, 45, 60, 65, 70, 80, 90]
}
df = pd.DataFrame(data)
df['pass'] = (df['marks'] >= 50).astype(int)

X = df[['hours']]
y = df['pass']

model = LogisticRegression()
model.fit(X, y)

# --- BUILD APP ---
st.title("Student Pass/Fail Predictor AI")
st.write("This is your first AI app!")

hours = st.slider("How many hours did you study?", 1, 10, 5)

if st.button("Predict"):
    prediction = model.predict([[hours]])[0]
    if prediction == 1:
        st.success(f"Studying {hours} hours: You will PASS!")
    else:
        st.error(f"Studying {hours} hours: You will FAIL. Study more!")