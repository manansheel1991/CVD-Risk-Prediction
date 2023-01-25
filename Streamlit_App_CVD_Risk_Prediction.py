import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title('Predict the risk of CVD (Cardiovascular Disease) in the next 10 years')

st.write('Enter the information for a person - ')

st.write('Select Gender -')
sex = st.selectbox('Gender', ('Male', 'Female'))
if sex == 'Male':
    sex = 1
elif sex == 'Female':
    sex = 0

age = st.slider('Enter Age', min_value=5, max_value=100)

st.write('Select Education Level - ')
education = st.selectbox('Education Level', (1, 2, 3, 4))

st.write('Select if the person is a Smoker - ')
currentSmoker = st.selectbox('Smoker', ('Yes', 'No'))
if currentSmoker == 'No':
    currentSmoker = 0
elif currentSmoker == 'Yes':
    currentSmoker = 1

st.write('Cigrattes smoked per day -')
cigsPerDay = st.number_input('Enter a number')

st.write('Select if the person has a Prevalent Stroke - ')
ps = st.selectbox('Stroke', ('Yes', 'No'))
if ps == 'No':
    prevalentStroke = 0
elif ps == 'Yes':
    prevalentStroke = 1

st.write('Select if the person has a Prevalent Hypertension - ')
ph = st.selectbox('Hypertension', ('Yes', 'No'))
if ph == 'No':
    prevalentHyp = 0
elif ph == 'Yes':
    prevalentHyp = 1

st.write('Select if the person has Diabetes - ')
db = st.selectbox('Diabetes', ('Yes', 'No'))
if db == 'No':
    diabetes = 0
elif db == 'Yes':
    diabetes = 1

st.write('Total Cholestrol -')
totChol = st.number_input('Enter cholosterol')

st.write('Enter Systolic Blood Pressure -')
sysBP = st.number_input('Enter SBP')

st.write('Enter Diabolic Blood Pressure -')
diaBP = st.number_input('Enter diaBP')

st.write('Enter BMI -')
BMI = st.number_input('Enter BMI')

st.write('Enter Heart Rate -')
heartRate = st.number_input('Enter Heart Rate')

st.write('Enter Glucose Level -')
glucose = st.number_input('Enter glucose level')

# ['sex', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'prevalentStroke',
#     'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

st.write('Select the algorithm -')
algo = st.selectbox('Algorithm', ('Logistic Regression', 'SVM', 'kNN'))

submit = st.button('Submit')

if submit:
    test_df = pd.DataFrame()

    # test_df['sex'].iloc[0] = sex
    # test_df['age'].iloc[0] = age
    # test_df['education'].iloc[0] = education
    # test_df['currentSmoker'].iloc[0] = currentSmoker
    # test_df['cigsPerDay'].iloc[0] = cigsPerDay
    # test_df['prevalentStroke'].iloc[0] = prevalentStroke
    # test_df['prevalentHyp'].iloc[0] = prevalentHyp
    # test_df['diabetes'].iloc[0] = diabetes
    # test_df['totalChol'].iloc[0] = totChol
    # test_df['sysBP'].iloc[0] = sysBP
    # test_df['diaBP'].iloc[0] = diaBP
    # test_df['BMI'].iloc[0] = BMI
    # test_df['heartRate'].iloc[0] = heartRate
    # test_df['glucose'].iloc[0] = glucose

    test_df = pd.DataFrame({'sex': sex, 'age': age, 'education': education, 'currentSmoker': currentSmoker, 'cigsPerDay': cigsPerDay, 'prevalentStroke': prevalentStroke,
                            'prevalentHype': prevalentHyp, 'diabetes': diabetes, 'totChol': totChol, 'sysBP': sysBP, 'diaBP': diaBP, 'BMI': BMI, 'heartRate': heartRate, 'glucose': glucose}, index=[0])

    print(test_df)

    if algo == 'Logistic Regression':
        with open('cvd_lr_model.pkl', 'rb') as f:
            clf = pickle.load(f)
    elif algo == 'SVM':
        with open('cvd_svc_model.pkl', 'rb') as f:
            clf = pickle.load(f)
    elif algo == 'kNN':
        with open('cvd_kn_model.pkl', 'rb') as f:
            clf = pickle.load(f)

    # test_df = np.asarray(test_df)

    result = clf.predict(test_df)

    if result == 0:
        st.success(
            'The person does not have the risk of CVD in the next 10 years')
    elif result == 1:
        st.error('The person has the risk of CVD in the next 10 years')
