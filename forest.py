import streamlit as st
import cv2
import tempfile
import folium
import time
from streamlit_folium import st_folium
from ultralytics import YOLO

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="🔥 Sentinel Command", layout="wide", page_icon="🚒")

# Custom CSS for the Red Alert Button
st.markdown("""
    <style>
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

# Header Section
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.title("🛰️ Sentinel: Forest Fire Defense System")
with col_header2:
    st.metric(label="System Status", value="ONLINE", delta="Active Monitoring")

st.markdown("---")

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') 

model = load_model()

# --- 3. DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 1]) 

# === LEFT COLUMN: SATELLITE MAP ===
with col1:
    st.subheader("🌍 Geospatial Live Feed")
    
    # Coordinates (Example: Bangalore) - CHANGE THIS TO YOUR CITY
    my_location = [12.9716, 77.5946]  
    
    # Create Map (Dark Mode for Sci-Fi look)
    m = folium.Map(location=my_location, zoom_start=13, tiles="CartoDB dark_matter")
    
    # Your Base Marker
    folium.Marker(
        location=my_location,
        popup="🏢 Command Center",
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(m)
    
    # Simulated Fire Location (Slightly offset from base)
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

    # Emergency Button
    st.markdown("### ⚠️ Emergency Controls")
    if st.button("🚨 DISPATCH FIRE RESPONSE TEAM 🚨"):
        with st.spinner("Contacting Local Fire Station..."):
            time.sleep(2) # Fake delay for realism
        
        st.success("✅ ALERT SENT! Units dispatched to Sector 4.")
        
        # Play Sound
        try:
            st.audio("alert.mp3", format="audio/mp3", start_time=0)
        except:
            pass


# === RIGHT COLUMN: DRONE AI VISION ===
with col2:
    st.subheader("🚁 Drone Reconnaissance")
    
    # Upload Option
    video_file = st.file_uploader("Upload Drone Footage", type=['mp4'])
    
    # Default to local file if nothing uploaded
    if not video_file:
        try:
            video_file = open("fire_video.mp4", "rb")
            st.info("System using automated patrol feed.")
        except:
            st.warning("⚠️ File 'fire_video.mp4' not found! Please add it to folder.")

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        # Live Stats Container
        stat_col1, stat_col2 = st.columns(2)
        kpi1 = stat_col1.metric("Confidence Level", "98%")
        kpi2 = stat_col2.metric("Objects Detected", "0")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # AI Processing
            results = model(frame)
            res_plotted = results[0].plot()
            
            # Update Object Counter
            detected_count = len(results[0].boxes)
            kpi2.metric("Objects Detected", f"{detected_count}")
            
            # Display Video
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, use_container_width=True)
            
        cap.release()