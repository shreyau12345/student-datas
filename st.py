import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student SOTY Predictor", layout="wide")

st.title("🎓 Student of the Year (SOTY) Predictor")

# File Upload
uploaded_file = st.file_uploader("Upload Student CSV File", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Features & Target
    X = df[['SP', 'Marks', 'Attendance', 'Age']]
    y = df['Soty']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"Accuracy: **{accuracy * 100:.2f}%**")

    st.subheader("Make a Prediction")

    col1, col2 = st.columns(2)

    with col1:
        sp = st.number_input("SP", min_value=0)
        marks = st.number_input("Marks", min_value=0)
    
    with col2:
        attendance = st.number_input("Attendance", min_value=0)
        age = st.number_input("Age", min_value=0)

    if st.button("Predict SOTY"):
        input_data = pd.DataFrame([[sp, marks, attendance, age]],
                                  columns=['SP', 'Marks', 'Attendance', 'Age'])
        
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("🏆 This student is likely SOTY!")
        else:
            st.error("❌ This student is not SOTY.")