import pickle
import streamlit as st
import numpy as np
import sklearn

st.header('Heart Disease Predictor', divider='rainbow')

with open('heart_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)



st.title('Heart Disease Predictor')
st.write(model)
input_data = (20, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

st.write(model.predict())
# Create input elements (e.g., sliders, text input fields) for users to enter data
age = st.slider('Age', 20, 100, 50)
# Add input elements for other features here...

# Make predictions when a user clicks a button
if st.button('Predict'):
    input_data = (20, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
    # input_data = (.2, 0, 0, .140, .268, 0, 0, .160, 0, .36, 0, .2, .2)
    input_data_as_numpy_array = np.asarray(input_data)
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    st.write(input_data_reshaped)
    prediction = model.predict(input_data_reshaped)

    # if prediction[0] == 0:
    #     st.write('The person does not have heart disease.')
    # else:
    #     st.write('The person has heart disease.')
