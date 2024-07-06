import streamlit as st
import pickle
import pandas as pd
import datetime

# loading model
with open('model/model.pkl','rb') as file:
    model_data = pickle.load(file)

df = model_data['data']
model = model_data['model']
preprocessor = model_data['preprocessor']

# title
st.title("Salary Prediction")


# DOJ 
doj = st.date_input("Date of Joining", format="DD/MM/YYYY")
doj_month = doj.month
doj_year = doj.year

# gender
gender = st.selectbox("Gender", df['SEX'].unique())

# designation
designation = st.selectbox("Designation", df['DESIGNATION'].unique())

# age
age = st.number_input("Age", min_value=0)

# unit
unit = st.selectbox("Department", df['UNIT'].unique())

# ratings
ratings = st.selectbox("Ratings", [0,1,2,3,4,5])

# past exp
past_exp = st.number_input("Past Experience (in Years)", min_value=0)

if st.button("Predict Salary"):
    # converting user input into dataframe
    user_input = {'SEX':gender,
                  'DESIGNATION':designation,
                  'AGE':age,
                  'UNIT':unit,
                  'RATINGS':ratings,
                  'PAST EXP':past_exp,
                  'MONTH':doj_month,
                  'YEAR':doj_year }
    input_df =pd.DataFrame([user_input])
    transform_input = preprocessor.transform(input_df)
    output_answer = model.predict(transform_input)[0]
    st.title(f'Salary will be around: {round(output_answer,2)}')
