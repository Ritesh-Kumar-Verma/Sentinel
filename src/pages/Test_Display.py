import streamlit as st
import cv2
import time


def display_rtsp_video(
    rtsp_url,
    width=700,
    height=430
):
    """
    Display an RTSP stream in Streamlit.

    Args:
        rtsp_url: RTSP camera URL
        width: Display width
        height: Display height
    """

    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        st.error("❌ Unable to connect to RTSP camera.")
        return

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

            # Display frame
            frame_placeholder.image(
                frame,
                width=width
            )

            # Small delay
            time.sleep(0.03)

    finally:
        cap.release()





display_rtsp_video(rtsp_url='rtsp://admin:admin@123@192.168.1.240:554/cam/realmonitor?channel=4&subtype=0')
