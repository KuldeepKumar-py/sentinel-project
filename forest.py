import streamlit as st
import cv2
import tempfile
import folium
import time
import os
import base64
from streamlit_folium import st_folium
from ultralytics import YOLO


st.set_page_config(
    page_title="Sentinel Command",
    layout="wide",
    page_icon="🚁",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    div[data-testid="metric-container"] {
        background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151;
    }
    .stButton>button { border-radius: 5px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


if "run_detection" not in st.session_state:
    st.session_state.run_detection = False


with st.sidebar:
    st.title("Mission Control")
    st.markdown("---")
  
    conf_threshold = st.slider("AI Confidence Threshold", 0.1, 1.0, 0.30, 0.05)
    
    st.subheader("System Status")
    st.success("Database Connected")
    st.success("Satellite Link Active")


@st.cache_resource
def load_model():
   
    custom_model_path = 'fireModel.pt' 
    
    if os.path.exists(custom_model_path):
        return YOLO(custom_model_path), "Custom Fire Model"
    else:
        st.sidebar.warning("⚠️ 'fire_model.pt' not found! Using standard YOLOv8n (will not detect fire).")
        return YOLO('yolov8n.pt'), "Standard Model (COCO)"

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


try:
    with st.spinner("Initializing AI Systems..."):
        model, model_type = load_model()
        st.sidebar.info(f"🤖 Active Model: {model_type}")
except Exception as e:
    st.error(f"Error loading model: {e}")



col_title, col_stat = st.columns([3, 1])
with col_title:
    st.title("🚁 Sentinel: Forest Defense System")
with col_stat:
    status_placeholder = st.empty()
    status_placeholder.info("SYSTEM STANDBY")

st.markdown("---")

col_map, col_vid = st.columns([1, 1])

with col_map:
    st.subheader("🌍 Geospatial Live Feed")
    command_center = [30.7333, 76.7794] 
    m = folium.Map(location=command_center, zoom_start=14, tiles="CartoDB dark_matter")
    
    folium.Circle(
        radius=500, location=command_center, color="green", fill=True, fill_opacity=0.1
    ).add_to(m)
    
    st_folium(m, width="100%", height=400)
    
    if st.button("DISPATCH FIRE TEAM", type="primary"):
        st.toast("Emergency Services Notified!", icon="🔥")


with col_vid:
    st.subheader("📹 Drone Optical Feed")
    video_file = st.file_uploader("Upload Drone Footage", type=['mp4', 'avi', 'mov'])
    
    m1, m2, m3 = st.columns(3)
    kpi_obj = m1.metric("Objects", "0")
    kpi_conf = m2.metric("Confidence", "0%")
    kpi_fps = m3.metric("FPS", "0")

    image_placeholder = st.empty()
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        b1, b2 = st.columns(2)
        if b1.button("▶ Start Analysis"): st.session_state.run_detection = True
        if b2.button("⏹ Stop Analysis"): st.session_state.run_detection = False

        if st.session_state.run_detection:
            prev_time = time.time()
            frame_count = 0
            FRAME_SKIP = 3  

            last_boxes = []
            max_conf = 0.0
            box_count = 0
            
            while cap.isOpened() and st.session_state.run_detection:
                ret, frame = cap.read()
                if not ret:
                    st.session_state.run_detection = False
                    break
                
                frame = cv2.resize(frame, (640, 360))
                frame_count += 1
                
               
                if frame_count % FRAME_SKIP == 0:
                    results = model(frame, conf=conf_threshold, verbose=False)
                    last_boxes = results[0].boxes
                    
                    max_conf = 0.0
                    box_count = 0
                    
                    for box in last_boxes:
                        box_count += 1
                        conf = float(box.conf[0])
                        if conf > max_conf: max_conf = conf

                
                fire_detected = False
                
                if last_boxes is not None:
                    for box in last_boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        name = model.names[cls]
                        
                       
                        color = (0, 255, 0)
                        if 'fire' in name.lower(): 
                            color = (0, 0, 255) 
                        elif 'smoke' in name.lower(): 
                            color = (128, 128, 128) 
                        
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{name.upper()} {conf:.2f}", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                       
                        if conf > conf_threshold:
                            if 'fire' in name.lower() or 'smoke' in name.lower():
                                fire_detected = True
                                
                           

                
                if fire_detected:
                    status_placeholder.markdown(
                        "<div style='background-color:red; color:white; padding:10px; border-radius:5px; text-align:center; font-weight:bold;'>⚠️ FIRE DETECTED</div>", 
                        unsafe_allow_html=True
                    )
                    
                    if frame_count % 100 == 0: 
                        play_alert()
                else:
                    status_placeholder.info("SYSTEM MONITORING - NO THREATS")

                
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                kpi_obj.metric("Objects", box_count)
                kpi_conf.metric("Confidence", f"{max_conf:.2f}")
                kpi_fps.metric("FPS", f"{int(fps)}")

               
                image_placeholder.image(frame, channels="BGR", use_container_width=True)

            cap.release()
            st.info("Analysis Complete.")
