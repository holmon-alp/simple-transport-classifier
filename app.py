import streamlit as st
from fastai.vision.all import *
import plotly.express as px
st.title("Transportlarni tanuvchi dastur")

types = ("png", "jpg", "gif", "svg", "webp")
st.text("Transport vositalarining turlarini aniqlovchi dastur")

img = st.file_uploader(label="Rasm yuklash", type=types)

if img:
    pil_img = PILImage.create(img)

    model = load_learner("transports.pkl")

    predict, pr_id, prob = model.predict(pil_img)
    st.image(pil_img)
    st.success(f"Bashorat: {predict}")
    st.info(f"Aniqlik: {(prob[pr_id] * 100):.2f}%")
    # print(model.dls.vocab, np.array(prob)*100 , sep='\n')
    fig = px.bar(x=model.dls.vocab, y=prob*100)
    st.plotly_chart(fig)