import streamlit as st
import cv2
import time
from PIL import Image
import os
import numpy as np
from pathlib import Path
from src.face_recognition.face_detector import FaceDetector
from src.face_recognition.face_encoder import FaceEncoder
from src.face_recognition.recognize import recognize_frame
from dotenv import load_dotenv

load_dotenv()



class Home_UI:
    
    def __init__(self,rstp_url) -> None:
        self.DATABASE_PATH = os.getenv('DATABASE_PATH')
        self.encoder = FaceEncoder()
        self.database = self.load_database()
        self.face_detector = FaceDetector()

        self.rstp_url = rstp_url
        st.set_page_config(
            page_title="Sentinel",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # ---------- CSS ----------
        st.markdown("""
        <style>
            .stApp {
                background: #080d12;
                color: #e8edf2;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            .block-container {
                padding: 1rem 1.5rem;
                max-width: 1500px;
            }

            .topbar {
                height: 58px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 18px;
                background: #101820;
                border: 1px solid #202b34;
                border-radius: 10px;
                margin-bottom: 14px;
            }

            .brand {
                font-size: 21px;
                font-weight: 700;
                color: #f3f7fa;
            }

            .status {
                color: #36d477;
                font-size: 14px;
                font-weight: 600;
            }

            [data-testid="stSidebar"] {
                background: #0c1218;
                border-right: 1px solid #202a32;
            }

            [data-testid="stSidebar"] > div:first-child {
                padding-top: 1rem;
            }

            .nav-title {
                color: #71808c;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin: 12px 0 8px;
            }

            .nav-item {
                padding: 10px 12px;
                margin: 3px 0;
                border-radius: 7px;
                color: #aeb9c2;
                font-size: 14px;
            }

            .nav-item.active {
                background: #18242d;
                color: #ffffff;
                border-left: 3px solid #3ddc84;
            }

            .card {
                background: #101820;
                border: 1px solid #202b34;
                border-radius: 10px;
                padding: 18px;
                height: 100%;
            }

            .card-title {
                font-size: 13px;
                color: #82909b;
                text-transform: uppercase;
                letter-spacing: .8px;
                margin-bottom: 14px;
            }

            .camera-box {
                height: 430px;
                background:
                    radial-gradient(
                        circle at 50% 45%,
                        #18242c 0%,
                        #0a1015 55%,
                        #060a0e 100%
                    );
                border: 1px solid #29343d;
                border-radius: 7px;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                overflow: hidden;
            }

            .camera-label {
                color: #687681;
                font-size: 14px;
                letter-spacing: 1px;
                z-index: 2;
            }

            .person {
                position: absolute;
                top: 125px;
                left: 50%;
                transform: translateX(-50%);
                text-align: center;
                font-size: 55px;
                filter: grayscale(.8);
                opacity: .85;
                z-index: 2;
            }

            .camera-grid {
                position: absolute;
                inset: 0;
                background-image:
                    linear-gradient(
                        rgba(255,255,255,.025) 1px,
                        transparent 1px
                    ),
                    linear-gradient(
                        90deg,
                        rgba(255,255,255,.025) 1px,
                        transparent 1px
                    );
                background-size: 40px 40px;
                z-index: 1;
            }

            .rec {
                color: #ff5353;
                font-weight: 700;
            }

            .camera-controls {
                display: flex;
                justify-content: space-between;
                padding: 12px 4px 4px;
                color: #7d8a94;
                font-size: 14px;
            }

            .camera-meta {
                color: #66747f;
                font-size: 12px;
                margin-top: 8px;
            }

            .person-name {
                font-size: 21px;
                font-weight: 700;
                margin-bottom: 7px;
            }

            .authorized {
                display: inline-block;
                background: rgba(54,212,119,.12);
                color: #36d477;
                border: 1px solid rgba(54,212,119,.25);
                border-radius: 20px;
                padding: 4px 9px;
                font-size: 12px;
            }

            .confidence {
                font-size: 30px;
                font-weight: 700;
                color: #eaf2f6;
                margin: 14px 0 20px;
            }

            .divider {
                border-top: 1px solid #263139;
                margin: 18px 0;
            }

            .recent-title {
                color: #83909a;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 10px;
            }

            .activity {
                display: flex;
                justify-content: space-between;
                padding: 9px 0;
                border-bottom: 1px solid #1c262e;
                font-size: 13px;
            }

            .activity:last-child {
                border-bottom: none;
            }

            .activity-name {
                color: #dce3e8;
            }

            .activity-time {
                color: #687681;
            }

            .unknown {
                color: #d5dce1;
            }

            .unknown-icon {
                color: #e5b84b;
            }

            .stButton button {
                background: #111b22;
                border: 1px solid #26333c;
                color: #aeb9c2;
                border-radius: 6px;
            }

            .stButton button:hover {
                border-color: #3ddc84;
                color: #ffffff;
            }
        </style>
        """, unsafe_allow_html=True)


        # ---------- Header ----------
        st.markdown("""
        <div class="topbar">
            <div class="brand">🛡️ Sentinel</div>
            <div class="status">🟢 SYSTEM ONLINE</div>
        </div>
        """, unsafe_allow_html=True)


        # ---------- Sidebar ----------
        with st.sidebar:

            st.markdown("### 🏠 Sentinel")

            st.markdown(
                '<div class="nav-title">Navigation</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="nav-item active">🏠 &nbsp; Dashboard</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="nav-item">📹 &nbsp; Cameras</div>',
                unsafe_allow_html=True
            )

            st.markdown("""
            <div style="
                padding-left:30px;
                color:#697781;
                font-size:12px;
                line-height:1.9;
            ">
                Front Door<br>
                Backyard<br>
                Garage
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                '<div class="nav-item">👤 &nbsp; People</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="nav-item">🚨 &nbsp; Alerts</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="nav-item">📜 &nbsp; Activity</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="nav-item">⚙️ &nbsp; Settings</div>',
                unsafe_allow_html=True
            )
        
        
        
        
        


        # ---------- Main layout ----------
        video_source = st.selectbox(
            "Video Source",
            ["Webcam", "RTSP"]
        )
        video_col, detection_col = st.columns(
            [2.15, 1],
            gap="medium"
        )
        
        # ---------- Camera ----------
        with video_col:
            if video_source == "Webcam":
                self.get_webcam_video()

            elif video_source == "RTSP":
                self.get_rtsp_video(rtsp_url=rstp_url)

        # ---------- Detection ----------
        with detection_col:

            st.markdown("""
            """, unsafe_allow_html=True)

        
    def get_rtsp_video(self,
        rtsp_url,
        width=700,
        height=430
    ):
        """
        Get an RTSP stream in Streamlit.

        Args:
            rtsp_url: RTSP camera URL
            width: Display width
            height: Display height
        """


        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            st.error("❌ Unable to connect to RTSP camera.")
            return

        frame_placeholder = st.empty()
        try:
            while True:

                ret, frame = cap.read()

                if not ret:
                    st.warning("⚠️ Unable to read video frame.")
                    break

                # Resize frame
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA
                )

                # Convert BGR → RGB
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
                
                frame_placeholder.image(frame)
                # Small delay
                time.sleep(0.03)

        finally:
            cap.release()
            
            
    
    def detect_face(self,img):


        faces = self.face_detector.detect(img)

        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            
            x1, y1, x2, y2 = bbox

            # print(
            #     f"Face {i + 1}: "
            #     f"({x1}, {y1}) -> ({x2}, {y2})"
            # )

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

        return img

    def load_database(self):
        """
        Load all face embeddings from the database.

        Returns:
            dict:
                {
                    "Name1": embedding,
                    "Name2": embedding
                }
        """

        database = {}

        for filename in os.listdir(self.DATABASE_PATH):
            if not filename.endswith(".npy"):
                continue

            person_name = os.path.splitext(filename)[0]

            path = os.path.join(
                self.DATABASE_PATH,
                filename,
            )

            embedding = np.load(path)

            database[person_name] = embedding

        return database



    def get_webcam_video(
        self,
        camera_index=0,
        width=700,
        height=430
    ):
        """
        Get webcam video in Streamlit.
        """

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            st.error("❌ Unable to connect to webcam.")
            return

        frame_placeholder = st.empty()

        frame_count = 0
        recognized_frame = None

        try:
            while True:

                ret, frame = cap.read()

                if not ret:
                    st.warning("⚠️ Unable to read webcam frame.")
                    break

                # Resize frame
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA
                )

                # Recognize every 5 frames
                if frame_count >= 29:

                    recognized_frame = recognize_frame(
                        frame,
                        database=self.database,
                        encoder=self.encoder
                    )

                    frame_count = 0

                else:
                    frame_count += 1

                # Use the last recognized frame
                if recognized_frame is not None:
                    display_frame = recognized_frame
                else:
                    display_frame = frame

                # BGR → RGB
                display_frame = cv2.cvtColor(
                    display_frame,
                    cv2.COLOR_BGR2RGB
                )

                frame_placeholder.image(
                    display_frame,
                    width=width
                )

                # time.sleep(0.03)

        finally:
            cap.release()
