import streamlit as st

import cv2

import tempfile

import folium

import time

import os

import base64

import streamlit.components.v1 as components 

from streamlit_folium import st_folium

from ultralytics import YOLO

from twilio.rest import Client



# ==========================================

# ⚙️ CONFIGURATION SECTION (EDIT THIS!)

# ==========================================



# 1. TWILIO SMS KEYS (Get these from twilio.com)

TWILIO_SID = "A"   # Example: "ACb..."

TWILIO_AUTH = ""  # Example: "a1b2..."

TWILIO_FROM = "+1"    # Example: "+1234567890"

TO_PHONE = "+91"    # Example: "+919876543210"



# 2. Page Setup

st.set_page_config(

    page_title="Sentinel Command",

    layout="wide",

    page_icon="🚁",

    initial_sidebar_state="expanded"

)



# ==========================================

# 🧠 HELPER FUNCTIONS

# ==========================================



# 1. Voice Alert Function (Text-to-Speech)

def speak(text):

    js = f"""

        <script>

            function speakText() {{

                var msg = new SpeechSynthesisUtterance("{text}");

                msg.rate = 1.0; 

                msg.pitch = 1.0; 

                msg.volume = 1.0; 

                var voices = window.speechSynthesis.getVoices();

                msg.voice = voices.find(v => v.name.includes('Google US English')) || voices[0];

                window.speechSynthesis.speak(msg);

            }}

            speakText();

        </script>

    """

    components.html(js, height=0, width=0)



# 2. SMS Alert Function

def send_sms(lat, long):

    try:

        # Only send if keys are actually set

        if "PASTE" in TWILIO_SID:

            st.error("❌ Twilio Keys missing! Check the top of 'forest.py'.")

            return False

            

        client = Client(TWILIO_SID, TWILIO_AUTH)

        message = client.messages.create(

            body=f"🔥 SENTINEL ALERT: Fire Detected!\n📍 Loc: {lat}, {long}\n🚁 Drone dispatching immediately.",

            from_=TWILIO_FROM,

            to=TO_PHONE

        )

        return True

    except Exception as e:

        st.error(f"SMS Failed: {e}")

        return False



# 3. Model Loading

@st.cache_resource

def load_model():

    custom_model_path = 'fireModel.pt'

    if os.path.exists(custom_model_path):

        return YOLO(custom_model_path), "Custom Fire Model"

    else:

        st.sidebar.error("⚠️ 'fireModel.pt' MISSING! Using Standard YOLO.")

        return YOLO('yolov8n.pt'), "Standard Model (COCO)"



# 4. Audio Siren

def play_alert(file_path="alert.mp3"):

    if os.path.exists(file_path):

        with open(file_path, "rb") as f:

            data = f.read()

            b64 = base64.b64encode(data).decode()

            md = f"""

                <audio autoplay="true">

                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">

                </audio>

                """

            st.markdown(md, unsafe_allow_html=True)



# ==========================================

# 🎨 UI & LAYOUT

# ==========================================



# Custom CSS

st.markdown("""

<style>

    .stApp { background-color: #0e1117; color: #e6e6e6; }

    div[data-testid="metric-container"] {

        background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151;

    }

    .fire-alert {

        color: #ff4b4b; font-weight: 900; font-size: 24px;

        animation: pulse 1s infinite; text-align: center;

        border: 2px solid #ff4b4b; padding: 10px; border-radius: 10px;

    }

    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

</style>

""", unsafe_allow_html=True)



# Sidebar

with st.sidebar:

    st.title("Mission Control")

    st.markdown("---")

    cam_index = st.selectbox("Select Camera Index", [0, 1, 2], index=0)

    conf_threshold = st.slider("AI Confidence Threshold", 0.1, 1.0, 0.35, 0.05)

    debug_mode = st.toggle("Debug Mode (Show All)", value=False)

    st.subheader("System Status")

    st.success("Database Connected")

    st.success("Satellite Link Active")



# Header

col_title, col_stat = st.columns([3, 1])

with col_title:

    st.title("🚁 Sentinel: Forest Defense System")

with col_stat:

    status_placeholder = st.empty()

    status_placeholder.info("SYSTEM STANDBY")



st.markdown("---")



# Main Interface

col_map, col_vid = st.columns([1, 1])



# --- LEFT COLUMN: MAP ---

with col_map:

    st.subheader("🌍 Geospatial Live Feed")

    command_center = [30.7333, 76.7794]

    

    # SATELLITE VIEW MAP

    m = folium.Map(

        location=command_center, 

        zoom_start=14, 

        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",

        attr="Esri World Imagery"

    )

    

    folium.Circle(

        radius=500, location=command_center, color="red", fill=True, fill_opacity=0.2

    ).add_to(m)

    

    st_folium(m, width="100%", height=400)

    

    # SMS DISPATCH BUTTON

    if st.button("DISPATCH FIRE TEAM", type="primary"):

        with st.spinner("Contacting Emergency Services..."):

            speak("Dispatching emergency units to sector 4.")

            time.sleep(1) 

            success = send_sms(30.7333, 76.7794)

            if success:

                st.toast("🚨 SMS SENT TO FIRE DEPT!", icon="✅")

                st.success(f"Message dispatched to: {TO_PHONE}")

            else:

                st.toast("Connection Failed", icon="❌")



# --- RIGHT COLUMN: DRONE FEED ---

with col_vid:

    st.subheader("📹 Drone Optical Feed")

    source_radio = st.radio("Select Source", ["📂 Upload Video", "🔴 Live Drone Feed"])

    

    cap = None

    if source_radio == "📂 Upload Video":

        video_file = st.file_uploader("Upload Drone Footage", type=['mp4', 'avi', 'mov'])

        if video_file:

            tfile = tempfile.NamedTemporaryFile(delete=False)

            tfile.write(video_file.read())

            cap = cv2.VideoCapture(tfile.name)

    elif source_radio == "🔴 Live Drone Feed":

        if st.checkbox("Activate Camera Link"):

            cap = cv2.VideoCapture(cam_index)



    # Metrics

    m1, m2, m3 = st.columns(3)

    kpi_obj = m1.metric("Objects", "0")

    kpi_conf = m2.metric("Confidence", "0%")

    kpi_fps = m3.metric("FPS", "0")

    image_placeholder = st.empty()

    

    # Initialize Session State

    if "run_detection" not in st.session_state:

        st.session_state.run_detection = False



    if cap is not None:

        b1, b2 = st.columns(2)

        if b1.button("▶ Start Analysis"): st.session_state.run_detection = True

        if b2.button("⏹ Stop Analysis"): st.session_state.run_detection = False



        if st.session_state.run_detection:

            # Initialize AI

            try:

                model, _ = load_model()

            except:

                st.error("Model Error")

                

            prev_time = time.time()

            frame_count = 0

            fire_streak = 0

            

            while cap.isOpened() and st.session_state.run_detection:

                ret, frame = cap.read()

                if not ret:

                    st.session_state.run_detection = False

                    break

                

                if source_radio == "🔴 Live Drone Feed":

                    frame = cv2.flip(frame, 1)



                frame = cv2.resize(frame, (640, 360))

                frame_count += 1

                

                # Inference (Skip frames for speed)

                if frame_count % 3 == 0:

                    results = model(frame, conf=conf_threshold, verbose=False)

                    boxes = results[0].boxes

                    

                    max_conf = 0.0

                    box_count = 0

                    fire_detected_now = False

                    

                    if boxes:

                        for box in boxes:

                            box_count += 1

                            conf = float(box.conf[0])

                            cls = int(box.cls[0])

                            name = model.names[cls]

                            if conf > max_conf: max_conf = conf

                            

                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            

                            # CHECK FOR FIRE

                            is_fire = 'fire' in name.lower() or 'smoke' in name.lower()

                            

                            if is_fire:

                                fire_detected_now = True

                                color = (0, 0, 255) # Red

                            else:

                                color = (255, 255, 255) # White

                            

                            # Draw Box

                            if is_fire or debug_mode:

                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                                cv2.putText(frame, f"{name.upper()} {conf:.2f}", (x1, y1-10),

                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



                    # Streak Logic

                    if fire_detected_now:

                        fire_streak += 1

                    else:

                        fire_streak = 0 # Instant reset to prevent false alarms

                    

                    # ALERT SYSTEM

                    if fire_streak >= 4:

                        status_placeholder.markdown(

                            "<div class='fire-alert'>⚠️ FIRE CONFIRMED</div>", 

                            unsafe_allow_html=True

                        )

                        # Voice & Sound Alert (Once every 100 frames)

                        if frame_count % 100 == 0:

                            play_alert()

                            speak("Critical Alert. Fire detected in sector 4.")

                            

                    elif fire_streak > 0:

                        status_placeholder.warning(f"⚠️ Analyzing... ({fire_streak}/4)")

                    else:

                        status_placeholder.info("SYSTEM MONITORING - NORMAL")

                    

                    # Update Metrics

                    fps = 1 / (time.time() - prev_time)

                    prev_time = time.time()

                    kpi_obj.metric("Objects", box_count)

                    kpi_conf.metric("Confidence", f"{max_conf:.2f}")

                    kpi_fps.metric("FPS", f"{int(fps)}")

                    

                    image_placeholder.image(frame, channels="BGR", use_container_width=True)



            cap.release()
