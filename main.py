# import cv2
# from src.face_recognition.face_encoder import FaceEncoder

# IMAGE_PATH = "test_images/Tony.png"


# def main():
#     img = cv2.imread(IMAGE_PATH)
#     encoder = FaceEncoder()
#     emb = encoder.encode(img)
#     print(emb.shape)
# if __name__ == "__main__":
#     main()


import streamlit as st
from src.pages.Test_Display import display_rtsp_video
from dotenv import load_dotenv
from src.pages.Home import Home_UI
import os
load_dotenv()

st.set_page_config(
    page_title="Sentinel",
    page_icon="🛡️",
    layout="wide"
)

home = Home_UI(rstp_url=os.getenv("RTSP_URL"))




# display_rtsp_video(os.getenv("RTSP_URL"))
