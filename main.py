import streamlit as st
from questionAnswering import answerQuestion

text = st.text_input("Enter some text 👇",
                     placeholder="What are some relevant experiences you've had?")

if text != '':
    stream = answerQuestion(text)
    answerStream = ''
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            answerStream += chunk.choices[0].delta.content
    st.write(answerStream)