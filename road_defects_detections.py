"""
RoadVision AI — All-in-one Streamlit app
Features:
 - Two YOLOv11 models (road markings + road defects)
 - Image / Video / Live webcam streaming (real-time-ish)
 - Batch processing (zip)
 - Folium geospatial mapping + heatmap + clusters
 - Live plots & dashboard (plotly)
 - Simple AI assistant (OpenAI integration placeholder)
 - Severity scoring + simple predictive trend (exponential smoothing)
 - Export: CSV / GeoJSON / ZIP / PDF report (simple)
 
 NOTES:
 - Put your YOLOv11 .pt files in weights/ and update MODEL_PATHS below if needed.
 - For AI assistant, set OPENAI_API_KEY in sidebar if you want GPT responses.
 - For realtime webcam, CPU will be slow; GPU recommended for >10 FPS.
"""

import json
import zipfile
import time
import tempfile
import os
import io
import cv2
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import folium
import plotly.graph_objects as go
import plotly.express as px
from collections import deque, defaultdict
from datetime import datetime
import base64
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st
st.set_page_config(page_title="RoadVision AI — Full Suite",
                   page_icon="🛣️", layout="wide")

# ---- Imports ----

# Visualizations

# Optional OpenAI usage
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---- CONFIG / PATHS ----
MODEL_PATHS = {
    "Road Markings (YOLOv11)": "road_markings_model.pt",
    "Road Defects (YOLOv11)": "road_defects_model.pt",
}

# Where to save exports
os.makedirs("outputs", exist_ok=True)

# ---- CACHING / MODEL LOAD ----


@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

# ---- Utility functions ----


def image_bytes_to_np(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def np_to_bytes(img_np, fmt="PNG"):
    im = Image.fromarray(img_np)
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


def save_image_file(img_np, outpath):
    Image.fromarray(img_np).save(outpath)


def timestamp():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# Basic severity scoring (customize as needed)


def severity_score(box, cls_name, conf, img_shape):
    # box: (x1,y1,x2,y2)
    x1, y1, x2, y2 = box
    area = max(0, (x2-x1)*(y2-y1))
    img_area = img_shape[0]*img_shape[1]
    rel_area = area / max(img_area, 1)
    # heuristics: defects larger and higher confidence -> higher severity
    base = rel_area * 100  # scale
    if "pothole" in cls_name.lower():
        base *= 2.5
    if "crack" in cls_name.lower():
        base *= 1.8
    score = (base * conf) * 10
    return min(100, score)


def boxes_from_result(result):
    # ultralytics result.boxes: tensor list -> convert to simple list
    boxes = []
    if hasattr(result, "boxes") and len(result.boxes) > 0:
        for b in result.boxes:
            try:
                # b.xyxy, b.conf, b.cls
                xy = b.xyxy[0].cpu().numpy() if hasattr(
                    b, "xyxy") else np.array([0, 0, 0, 0])
                conf = float(b.conf[0]) if hasattr(
                    b, "conf") else float(b.conf)
                cls = int(b.cls[0]) if hasattr(b, "cls") else int(b.cls)
            except Exception:
                # Fallback generic parsing
                try:
                    arr = b.cpu().numpy()
                    xy = arr[:4]
                    conf = float(arr[4])
                    cls = int(arr[5])
                except Exception:
                    continue
            boxes.append((float(xy[0]), float(xy[1]),
                         float(xy[2]), float(xy[3]), conf, cls))
    return boxes


def result_to_geojson(detections):
    # detections: list of dicts with lat,lng,... -> returns GeoJSON FeatureCollection
    features = []
    for d in detections:
        features.append({
            "type": "Feature",
            "properties": {k: v for k, v in d.items() if k not in ("lat", "lng")},
            "geometry": {"type": "Point", "coordinates": [d["lng"], d["lat"]]}
        })
    return {"type": "FeatureCollection", "features": features}


def download_button_bytes(data: bytes, file_name: str, mime: str):
    st.download_button(
        label=f"Download {file_name}", data=data, file_name=file_name, mime=mime)

# Small exponential smoothing predictor for counts


def simple_exp_smooth(series, alpha=0.4):
    if len(series) == 0:
        return []
    s = series[0]
    preds = []
    for x in series:
        s = alpha*x + (1-alpha)*s
        preds.append(s)
    return preds


# ---- Sidebar: model selection, OpenAI key, global settings ----
st.sidebar.header("Settings & Models")
selected_model = st.sidebar.selectbox(
    "Primary model (switch to run either)", list(MODEL_PATHS.keys()))
other_model = st.sidebar.selectbox("Secondary model (for quick switching)", list(
    MODEL_PATHS.keys()), index=1 if list(MODEL_PATHS.keys())[0] == list(MODEL_PATHS.keys())[1] else 0)
conf_thresh = st.sidebar.slider(
    "Confidence threshold", 0.1, 1.0, 0.4, step=0.05)
show_heatmap = st.sidebar.checkbox("Show heatmap on map", value=True)
use_gps = st.sidebar.checkbox("Attach GPS (simulate / extract)", value=False)
openai_key_input = st.sidebar.text_input(
    "OpenAI API Key (optional)", type="password")
if openai_key_input:
    os.environ["OPENAI_API_KEY"] = openai_key_input
    if OPENAI_AVAILABLE:
        openai.api_key = openai_key_input

# Load both models cached
model_primary = load_yolo_model(MODEL_PATHS[selected_model])
model_secondary = load_yolo_model(MODEL_PATHS[other_model])

# ---- App Layout ----
st.markdown("""
# 🛣️ RoadVision AI — Live Inspection Suite
**All-in-one**: detection (two models), live demo, mapping, analytics, AI assistant, and reporting.
""")

# Top-level tabs
tabs = st.tabs(["Demo & Live", "Map & Geo", "Dashboard & Plots",
               "Batch / Export", "AI Assistant", "About & Tips"])

# Shared in-memory store for detections during session
if "detections" not in st.session_state:
    # list of dicts: {ts,class,conf,x1,y1,x2,y2,lat,lng,severity,img_name}
    st.session_state["detections"] = []

# ---- TAB: Demo & Live ----
with tabs[0]:
    st.header("Live Demo & Inputs")
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        mode = st.radio("Input mode", [
                        "Upload Image", "Upload Video", "Webcam Stream", "Sample (preloaded)"])
        if mode == "Upload Image":
            up = st.file_uploader("Upload an image", type=[
                                  "jpg", "jpeg", "png"])
            if up:
                img_np = image_bytes_to_np(up)
                st.image(img_np, caption="Input image", use_column_width=True)
                if st.button("Run detection on image"):
                    t0 = time.time()
                    res = model_primary.predict(
                        img_np, conf=conf_thresh, verbose=False)
                    res0 = res[0]
                    annotated = res0.plot()  # RGB np
                    st.image(annotated, caption="Detections",
                             use_column_width=True)
                    # Extract boxes and store detections
                    names = res0.names if hasattr(res0, "names") else {}
                    boxes = boxes_from_result(res0)
                    for (x1, y1, x2, y2, conf, cls) in boxes:
                        cls_name = names.get(cls, str(cls))
                        lat = None
                        lng = None
                        if use_gps:
                            # simulate or extract: here we create simulated coords near Accra center (for demo)
                            lat = 5.6037 + (np.random.rand()-0.5)*0.02
                            lng = -0.1870 + (np.random.rand()-0.5)*0.02
                        sev = severity_score(
                            (x1, y1, x2, y2), cls_name, conf, img_np.shape)
                        st.session_state["detections"].append({
                            "ts": timestamp(), "class": cls_name, "conf": float(conf),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "lat": lat, "lng": lng, "severity": float(sev),
                            "image": getattr(up, "name", "uploaded_image")
                        })
                    st.success(
                        f"Found {len(boxes)} objects in this image. Took {time.time()-t0:.2f}s")
        elif mode == "Upload Video":
            v = st.file_uploader("Upload short MP4 (<=1min recommended)", type=[
                                 "mp4", "mov", "avi"])
            if v:
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.write(v.read())
                tmp.flush()
                st.video(tmp.name)
                if st.button("Process Video"):
                    st.info(
                        "Processing video — each frame will be sent through YOLO (CPU may be slow).")
                    cap = cv2.VideoCapture(tmp.name)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    p = st.progress(0)
                    out_path = tmp.name + "_out.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = cap.get(cv2.CAP_PROP_FPS) or 15
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    i = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = model_primary.predict(
                            rgb, conf=conf_thresh, verbose=False)
                        annotated = res[0].plot()
                        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                        writer.write(bgr)
                        # optionally store aggregated detections per frame (omitted for speed)
                        i += 1
                        if total > 0:
                            p.progress(min(i/total, 1.0))
                    writer.release()
                    cap.release()
                    st.success("Video processed")
                    st.video(out_path)
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "Download processed video", f, file_name="processed_video.mp4")
        elif mode == "Webcam Stream":
            st.info("Start webcam stream — toggle Stop to end. NOTE: real-time speed depends on your machine and model size. GPU recommended.")
            start_stream = st.button("Start Stream")
            stop_stream = st.button("Stop Stream")
            stream_placeholder = st.empty()
            # Use session flag
            if "streaming" not in st.session_state:
                st.session_state.streaming = False
            if start_stream:
                st.session_state.streaming = True
            if stop_stream:
                st.session_state.streaming = False
            if st.session_state.streaming:
                cap = cv2.VideoCapture(0)
                fps_count = deque(maxlen=30)
                last = time.time()
                while st.session_state.streaming:
                    ret, frame = cap.read()
                    if not ret:
                        stream_placeholder.info("Cannot read from webcam.")
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = model_primary.predict(
                        rgb, conf=conf_thresh, verbose=False)
                    annotated = res[0].plot()
                    stream_placeholder.image(
                        annotated, caption="Live detection", use_column_width=True)
                    # store detections with simulated gps if enabled
                    boxes = boxes_from_result(res[0])
                    for (x1, y1, x2, y2, conf, cls) in boxes:
                        cls_name = res[0].names.get(cls, str(cls))
                        lat = None
                        lng = None
                        if use_gps:
                            lat = 5.6037 + (np.random.rand()-0.5)*0.02
                            lng = -0.1870 + (np.random.rand()-0.5)*0.02
                        sev = severity_score(
                            (x1, y1, x2, y2), cls_name, conf, rgb.shape)
                        st.session_state["detections"].append({
                            "ts": timestamp(), "class": cls_name, "conf": float(conf),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "lat": lat, "lng": lng, "severity": float(sev), "image": "webcam_frame"
                        })
                    # simple fps display & small sleep
                    now = time.time()
                    fps_count.append(1.0 / max((now-last), 1e-6))
                    last = now
                    avg_fps = sum(fps_count) / \
                        len(fps_count) if len(fps_count) > 0 else 0.0
                    st.metric("Live FPS (approx)", f"{avg_fps:.1f}")
                    # break if user stops via button - Streamlit will rerun; we check flag each loop
                cap.release()
                st.success("Stream stopped")
        else:
            # sample mode - show some bundled sample image(s)
            st.info("Sample images included for quick demo.")
            sample_path = "sample_data"
            # create sample images placeholder if not exist
            if not os.path.exists(sample_path):
                os.makedirs(sample_path, exist_ok=True)
                # create a dummy blank image for display
                blank = np.full((480, 640, 3), 220, dtype=np.uint8)
                save_image_file(blank, os.path.join(
                    sample_path, "blank_sample.png"))
            samples = [os.path.join(sample_path, f) for f in os.listdir(
                sample_path) if f.lower().endswith((".png", ".jpg"))]
            choice = st.selectbox("Pick sample", samples)
            if choice:
                img_np = cv2.cvtColor(cv2.imread(choice), cv2.COLOR_BGR2RGB)
                st.image(img_np, use_column_width=True)
                if st.button("Run detection on sample"):
                    res = model_primary.predict(
                        img_np, conf=conf_thresh, verbose=False)
                    st.image(res[0].plot(), use_column_width=True)

    with col2:
        st.markdown("### Session Info")
        st.write(f"Primary model: **{selected_model}**")
        st.write(f"Secondary model: **{other_model}**")
        st.write(f"Confidence threshold: **{conf_thresh:.2f}**")
        st.write(
            f"Stored detections in session: **{len(st.session_state['detections'])}**")
        if st.button("Clear stored detections"):
            st.session_state["detections"] = []
            st.success("Cleared")
        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("Export detections CSV"):
            df = pd.DataFrame(st.session_state["detections"])
            csv = df.to_csv(index=False).encode("utf-8")
            download_button_bytes(csv, "detections.csv", "text/csv")
        if st.button("Export GeoJSON"):
            dets_with_coords = [
                d for d in st.session_state["detections"] if d.get("lat") is not None]
            if len(dets_with_coords) == 0:
                st.warning("No detections with coordinates to export.")
            else:
                gj = result_to_geojson(dets_with_coords)
                data = json.dumps(gj).encode("utf-8")
                download_button_bytes(
                    data, "detections.geojson", "application/geo+json")

# ---- TAB: Map & Geo ----
with tabs[1]:
    st.header("Geospatial Map — Hotspots & Clusters")
    map_col, ctrl_col = st.columns([0.75, 0.25])
    with ctrl_col:
        st.write("Map Controls")
        cluster = st.checkbox("Use cluster markers", value=True)
        heat = st.checkbox("Show heatmap", value=show_heatmap)
        center_lat = st.number_input(
            "Map center lat", value=5.6037, format="%.6f")
        center_lng = st.number_input(
            "Map center lng", value=-0.1870, format="%.6f")
        zoom = st.slider("Zoom", 6, 18, 13)
        if st.button("Clear detections (with coords)"):
            st.session_state["detections"] = [
                d for d in st.session_state["detections"] if d.get("lat") is None]
            st.success("Cleared geotagged detections")

    with map_col:
        m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom)
        dets = [d for d in st.session_state["detections"]
                if d.get("lat") is not None]
        if cluster:
            mc = MarkerCluster()
            for d in dets:
                popup = f"Class: {d['class']}<br>Conf: {d['conf']:.2f}<br>Severity: {d['severity']:.1f}<br>Time: {d['ts']}"
                folium.Marker([d["lat"], d["lng"]], popup=popup).add_to(mc)
            m.add_child(mc)
        else:
            for d in dets:
                folium.CircleMarker(location=[d["lat"], d["lng"]], radius=6, color="red" if d["severity"]
                                    > 60 else "orange", popup=f"{d['class']} ({d['conf']:.2f})").add_to(m)
        if heat and len(dets) > 0:
            heat_data = [[d["lat"], d["lng"], d["severity"]] for d in dets]
            HeatMap(heat_data, radius=25, blur=15, max_zoom=14).add_to(m)
        # show map
        st_folium(m, width=900, height=600)

# ---- TAB: Dashboard & Plots ----
with tabs[2]:
    st.header("Analytics Dashboard")
    dets = pd.DataFrame(st.session_state["detections"])
    if dets.empty:
        st.info("No detections yet — run some detections to populate charts.")
    else:
        # summary stats
        st.subheader("Summary")
        colA, colB, colC = st.columns(3)
        colA.metric("Total Detections", len(dets))
        colB.metric("Geotagged", len(dets[dets['lat'].notnull()]))
        avg_sev = dets['severity'].mean() if 'severity' in dets.columns else 0
        colC.metric("Avg Severity", f"{avg_sev:.1f}")

        st.subheader("Class distribution")
        if 'class' in dets.columns:
            class_count = dets['class'].value_counts().reset_index()
            class_count.columns = ['class', 'count']
            fig = px.pie(class_count, names='class',
                         values='count', title='Detections by Class')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detections over time")
        # convert ts to datetime if present
        if 'ts' in dets.columns:
            dets['ts_dt'] = pd.to_datetime(dets['ts'])
            timeseries = dets.set_index('ts_dt').resample(
                '1T').size().rename("count").reset_index()
            fig2 = px.line(timeseries, x='ts_dt', y='count',
                           title='Detections per minute')
            st.plotly_chart(fig2, use_container_width=True)

            # predictive simple smoothing
            counts = list(timeseries['count'])
            preds = simple_exp_smooth(counts, alpha=0.4)
            if len(preds) > 0:
                fig2.add_scatter(
                    x=timeseries['ts_dt'], y=preds, mode='lines', name='Smoothed')
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Severity distribution")
        if 'severity' in dets.columns:
            fig3 = px.histogram(dets, x='severity', nbins=20,
                                title='Severity histogram')
            st.plotly_chart(fig3, use_container_width=True)

# ---- TAB: Batch / Export ----
with tabs[3]:
    st.header("Batch Processing & Reports")
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        zip_file = st.file_uploader(
            "Upload ZIP of images for batch detection", type=["zip"])
        if zip_file:
            if st.button("Run batch detection"):
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.write(zip_file.read())
                tmp.flush()
                outzip = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".zip")
                with zipfile.ZipFile(tmp.name, "r") as zin, zipfile.ZipFile(outzip.name, "w") as zout:
                    image_names = [f for f in zin.namelist(
                    ) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    p = st.progress(0)
                    for i, name in enumerate(image_names):
                        data = zin.read(name)
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        np_img = np.array(img)
                        res = model_secondary.predict(
                            np_img, conf=conf_thresh, verbose=False)
                        ann = res[0].plot()
                        out_bytes = np_to_bytes(ann)
                        zout.writestr(
                            f"detected_{os.path.basename(name)}", out_bytes)
                        # also append detections to session (no coords)
                        boxes = boxes_from_result(res[0])
                        for (x1, y1, x2, y2, conf, cls) in boxes:
                            cls_name = res[0].names.get(cls, str(cls))
                            sev = severity_score(
                                (x1, y1, x2, y2), cls_name, conf, np_img.shape)
                            st.session_state["detections"].append({
                                "ts": timestamp(), "class": cls_name, "conf": float(conf),
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "lat": None, "lng": None, "severity": float(sev),
                                "image": os.path.basename(name)
                            })
                        p.progress((i+1)/len(image_names))
                st.success("Batch complete")
                with open(outzip.name, "rb") as f:
                    st.download_button(
                        "Download batch results zip", f, file_name="batch_results.zip")
    with col2:
        st.subheader("Report Generation")
        if st.button("Generate quick PDF report (summary)"):
            # lightweight PDF generation using HTML -> PDF is heavy; we provide CSV + simple HTML
            df = pd.DataFrame(st.session_state["detections"])
            if df.empty:
                st.warning("No detections to report.")
            else:
                html = df.to_html(index=False)
                html_path = os.path.join(
                    "outputs", f"report_{int(time.time())}.html")
                with open(html_path, "w", encoding="utf-8") as fh:
                    fh.write(
                        f"<h1>RoadVision Quick Report</h1><p>Generated: {datetime.utcnow()}</p>{html}")
                with open(html_path, "rb") as f:
                    st.download_button("Download HTML report", f, file_name=os.path.basename(
                        html_path), mime="text/html")
                st.success(
                    "HTML report generated. For PDF, convert HTML to PDF externally or use wkhtmltopdf in deployment.")

# ---- TAB: AI Assistant ----
with tabs[4]:
    st.header("AI Assistant (Ask about detections & summary)")
    st.markdown("You can ask the assistant about the dataset, request summaries, or ask how to improve models. If you provided an OpenAI key in sidebar, responses will be generated via OpenAI. Otherwise, a simple local rule-based assistant will respond.")
    user_q = st.text_input(
        "Ask something to the assistant (e.g., 'Summarize detections', 'What roads need urgent repairs?')")

    if st.button("Ask Assistant") and user_q.strip() != "":
        df = pd.DataFrame(st.session_state["detections"])
        # Build context
        summary = ""
        if not df.empty:
            top_classes = df['class'].value_counts().head(5).to_dict()
            avg_sev = df['severity'].mean()
            geocount = df['lat'].count() if 'lat' in df.columns else 0
            summary = f"Total detections: {len(df)}. Top classes: {top_classes}. Average severity: {avg_sev:.1f}. Geotagged: {geocount}."
        else:
            summary = "No detections yet."

        prompt = f"""You are RoadVision Assistant. Context: {summary}\nUser question: {user_q}\nProvide a concise helpful answer and actionable next steps."""
        reply = None
        if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",  # placeholder - change as desired
                    messages=[{"role": "system", "content": "You are a helpful assistant for road defect detection."},
                              {"role": "user", "content": prompt}],
                    max_tokens=250
                )
                reply = resp['choices'][0]['message']['content']
            except Exception as e:
                reply = f"(OpenAI API error) {e}\n\nFallback summary:\n{summary}"
        else:
            # simple rule-based fallback
            if "summar" in user_q.lower():
                reply = "Quick summary: " + summary + \
                    " Next steps: collect more geotagged samples, validate top classes, prioritize segments with severity>60."
            elif "repair" in user_q.lower() or "urgent" in user_q.lower():
                if not df.empty:
                    high = df[df['severity'] > 60]
                    n = len(high)
                    reply = f"There are {n} high-severity detections. Recommend field inspection and fast patches for these segments."
                else:
                    reply = "No detections to analyze. Run detections first."
            else:
                reply = "I can summarize detections, list urgent items, suggest data collection tips, or propose model improvements. Try: 'Summarize detections' or 'What needs urgent repair?'"

        st.markdown("**Assistant:**")
        st.write(reply)

# ---- TAB: About & Tips ----
with tabs[5]:
    st.header("About, Tips & Deployment")
    st.markdown("""
    **Tips**
    - Use GPU for real-time webcam processing (NVIDIA CUDA & ultralytics will use GPU automatically).
    - For production, serve model inference in a dedicated API (FastAPI + ONNXRuntime/TensorRT) and keep Streamlit as frontend.
    - Collect geotagged frames for accurate mapping; many phones/inspections provide GPS metadata.
    - To increase speed: quantize model, use smaller backbone, or process every nth frame.
    """)
    st.markdown("**Where to edit model paths**")
    st.code("MODEL_PATHS = {\n    'Road Markings (YOLOv11)': 'weights/road_markings_yolo11.pt',\n    'Road Defects (YOLOv11)': 'weights/road_defects_yolo11.pt',\n}", language="python")
    st.markdown(
        "**Deployment**: Use Docker + GPU VM (AWS/GCP/Azure) or Streamlit Cloud for demos (no GPU).")
    st.markdown("---")
    st.markdown("Made with ❤️ by Vicentia — Good luck at the competition!")
