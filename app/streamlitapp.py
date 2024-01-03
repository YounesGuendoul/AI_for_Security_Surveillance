import streamlit as st
import os 
import imageio 
import tensorflow as tf 
from helpfnct import load_data, num_to_char
from modelayout import load_model

st.set_page_config(layout='wide')

st.markdown("""
    <style>
    .title {
        text-align: center;                 
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Renforcement de la communication : un outil d'IA pour les personnes sourdes</h1>", unsafe_allow_html=True) 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)


if options: 
    file_path = os.path.join('..','data','s1', selected_video)
    st.info('This is all the machine learning model sees when making a prediction')
    video, annotations = load_data(tf.convert_to_tensor(file_path))
    imageio.mimsave('animation.gif', video, fps=10)
    st.image('animation.gif', width=600) 

    st.info('This is the output of the machine learning model as tokens')
    model = load_model()
    yhat = model.predict(tf.expand_dims(video, axis=0))
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    st.text(decoder)

    st.info('Decode the raw tokens into words')
    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
    st.text(converted_prediction)
    
