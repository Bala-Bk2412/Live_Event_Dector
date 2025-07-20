import cv2
import streamlit as st
import time
from transformers import BlipProcessor,BlipForQuestionAnswering

preprocess = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cpu")

def work(image,question):
    inputs = preprocess(image, question, return_tensors="pt").to("cpu")
    out = model.generate(**inputs)
    result = preprocess.decode(out[0], skip_special_tokens=True)
    return str(result)



st.title("Webcam Live Feed")
FRAME_WINDOW = st.image([])
question = st.text_input("Enter the question:")
run = st.button('start stream')
stop = st.button("stop")
st.write("Output:")
camera = cv2.VideoCapture(0)
place_holder = st.empty()

if run:
    frame_count = 0
    while True:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        if frame_count % 10 == 0:
            out = work(frame, question)
        frame_count+=1
        place_holder.write(out)
        if stop:
            break
camera.release()
cv2.destroyAllWindows()