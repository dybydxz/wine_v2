import streamlit as st
import pandas as pd
import pickle 
import sklearn
#from sklearn.preprocessing import QuantileTransformer


st.sidebar.title('Input')
st.sidebar.image('www.jpg')


def get_input():
    
     alc = st.sidebar.slider('alcohol', 8.4, 14.9, 13.0)
     sulp = st.sidebar.slider('sulphates', 0.33, 2.0, 1.33)
     C_acid = st.sidebar.slider('citric acid', 0.00,1.00,0.55)    
     F_acid = st.sidebar.slider('fixed acidity', 4.6, 15.9, 6.0)
     res_sugar = st.sidebar.slider('residual sugar', 0.9, 15.5, 7.0)
    
    
     V_acid = st.sidebar.slider('volatile acidity', 0.12, 1.58, 0.80)
     total_sd = st.sidebar.slider('total sulfur dioxide', 6, 289, 100)
     des = st.sidebar.slider('density', 0.99007, 1.00369, 0.99999)      
     chlor = st.sidebar.slider('chlorides', 0.012, 0.611, 0.300)
     free_sd = st.sidebar.slider('free sulfur dioxide', 1, 72, 30)
     ph = st.sidebar.slider('pH', 2.74, 4.01, 3.00)
     
     

     data = {'alcohol': alc,
             'sulphates': sulp,
             'citric acid': C_acid,
             'fixed acidity': F_acid,
             'residual sugar': res_sugar,
             'volatile acidity': V_acid,
             'total sulfur dioxide': total_sd,
             'density': des,
             'chlorides': chlor,
             'free sulfur dioxide': free_sd,
             'pH': ph   
             
            }

     data_df = pd.DataFrame(data, index=[0])
     return data_df

# -- Call function to display widgets and get data from user
df = get_input()
    
st.title('Red Wine Quarity Prediction')

audio_file = open('music.ogg', 'rb')
audio_bytes = audio_file.read()

st.audio(audio_bytes, format='audio/ogg')

# -- Display new data from user inputs:
st.subheader('User Input Data :')
st.write(df)


# -- Reads the saved normalization model
load_sc = pickle.load(open('nor.pkl', 'rb'))
#Apply the normalization model to new data
df = load_sc.transform(df)

# -- Display normalized new data:
st.subheader('Normalized :')
st.write(df)

# -- Reads the saved classification model
load_knn = pickle.load(open('knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(df)

# -- Display predicted class:
st.subheader('Prediction :')
#penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(prediction)



st.image('wine3.gif')



