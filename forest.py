import streamlit as st
import cv2
import tempfile
import folium
import time
from streamlit_folium import st_folium
from ultralytics import YOLO


st.set_page_config(page_title="🔥 Sentinel Command", layout="wide", page_icon="🚁")


st.markdown("""
    <style>
    /* Import Google Font to match your design */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    /* --- STYLE THE FILE UPLOADER TO MATCH YOUR HTML --- */
    [data-testid='stFileUploader'] {
        width: 100%;
    }

    /* The Dropzone Box */
    [data-testid='stFileUploader'] section {
        background-color: #f8fafc; /* Your light bg color */
        border: 2px dashed #cbd5e1; /* Your border color */
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Hover Effect */
    [data-testid='stFileUploader'] section:hover {
        border-color: #3b82f6; /* Recon Blue */
        background-color: #eff6ff;
    }

    /* The "Browse Files" Button inside */
    [data-testid='stFileUploader'] button {
        background-color: #3b82f6; /* Recon Blue */
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 600;
    }
    
    /* The Red Alert Button (Keep this from before) */
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-color: #ff4b4b;
        font-size: 20px;
        height: 3em;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.title("🚁 Sentinel: Drone Reconnaissance System")
with col_header2:
    st.metric(label="System Status", value="ONLINE", delta="Active Monitoring")

st.markdown("---")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') 

model = load_model()

col1, col2 = st.columns([1, 1]) 


with col1:
    st.subheader("🌍 Geospatial Live Feed")
    my_location = [12.9716, 77.5946]  # Example: Bangalore
    
    m = folium.Map(location=my_location, zoom_start=13, tiles="CartoDB dark_matter")
    
    folium.Marker(
        location=my_location,
        popup="🏢 Command Center",
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(m)
    
    fire_loc = [my_location[0] + 0.01, my_location[1] + 0.01]
    folium.CircleMarker(
        location=fire_loc,
        radius=20,
        color="red",
        fill=True,
        fill_color="red",
        popup="🔥 CRITICAL FIRE ZONE"
    ).add_to(m)

    st_folium(m, width=700, height=450)

    st.markdown("### ⚠️ Emergency Controls")
    if st.button("🚨 DISPATCH FIRE RESPONSE TEAM 🚨"):
        with st.spinner("Contacting Local Fire Station..."):
            time.sleep(2)
        st.success("✅ ALERT SENT! Units dispatched.")
        try:
            st.audio("alert.mp3", format="audio/mp3", start_time=0)
        except:
            pass


with col2:
   
    st.markdown("""
        <div style="text-align: center; margin-bottom: 10px;">
            <span style="font-size: 2rem;">🚁</span>
            <h2 style="display: inline-block; margin-left: 10px;">Drone Feed</h2>
        </div>
    """, unsafe_allow_html=True)
    
    
    video_file = st.file_uploader("Upload Drone Footage", type=['mp4'])
    
    
    st.caption("Limit 200MB per file • MP4, MPEG4")
    
    if not video_file:
        try:
            video_file = open("fire_video.mp4", "rb")
            st.info("Using automated patrol feed.")
        except:
            st.warning("⚠️ File 'fire_video.mp4' not found!")

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        stat_col1, stat_col2 = st.columns(2)
        kpi1 = stat_col1.metric("Confidence Level", "98%")
        kpi2 = stat_col2.metric("Objects Detected", "0")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)
            res_plotted = results[0].plot()
            detected_count = len(results[0].boxes)
            kpi2.metric("Objects Detected", f"{detected_count}")
            
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, use_container_width=True)
            
        cap.release()
