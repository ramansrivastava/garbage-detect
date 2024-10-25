import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import supervision as sv
import cv2
import tempfile
import os
from pathlib import Path
import numpy as np
from io import BytesIO

# Initialize annotators (using the updated BoxAnnotator)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Inject Bootstrap into the Streamlit app
st.markdown("""
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f4fff4;  /* Light green background */
            color: #4CAF50;  /* Green font color */
        }
        .stApp {
            background-color: #ffffff;  /* White background for main app */
        }
        .btn-primary, .stButton>button {
            background-color: #FFFFFF !important;  /* Green buttons */
            border-color: #4CAF50 !important;
            color: white !important;  /* White text on green buttons */
        }
        .btn-primary:hover, .stButton>button:hover {
            background-color: #45a049 !important;  /* Darker green on hover */
            border-color: #45a049 !important;
            color: white !important;
        }
        .container {
            background-color: #f4fff4;  /* Light green containers */
            padding: 20px;
            border-radius: 10px;
        }
        .stSidebar .css-1dq8tca, .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p {
            color: #4CAF50;  /* Green text in the sidebar */
        }
        .stProgress .st-dl {
            background-color: #4CAF50 !important;  /* Green progress bar */
        }
        table {
            color: #000000 !important;  /* Black text for table */
            background-color: #ffffff !important;  /* White background for table */
        }
        table th, table td {
            color: #000000 !important;  /* Ensure black text in both header and data cells */
        }
        h1, h2, h3, h4, h5, h6, p, label {
            color: #4CAF50 !important;  /* Green text for all headers, paragraphs, labels */
        }
        .stMarkdown {
            color: #4CAF50 !important;  /* Green text for markdown */
        }
        input, textarea, select {
            color: #4CAF50 !important;  /* Green text for input fields */
            border-color: #4CAF50 !important;  /* Green borders for inputs */
        }
        input:focus, textarea:focus, select:focus {
            box-shadow: 0 0 5px #4CAF50 !important;  /* Green glow on focus */
        }
        .stProgress .st-fx {
            background-color: #4CAF50 !important;  /* Green progress bar foreground */
        }
    </style>
    """, unsafe_allow_html=True)




st.title("CleanSweep")
st.sidebar.header("Model Config")

# Model configuration
confidence = st.sidebar.slider(
    "Select Model Confidence", 
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05
)

# Source type selection
source_type = st.sidebar.radio("Select Source Type", ["Image", "Video"])
model_choice = st.sidebar.radio("Select Model", ["Model"])

# Load YOLO model
@st.cache_resource
def main_model():
    model = YOLO('best.pt')
    return model

def process_uploaded_video(video_bytes):
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_bytes)
        video_path = tmpfile.name

    try:
        # Read the video
        cap = cv2.VideoCapture(video_path)        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a temporary file for output
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_tmpfile:
            output_path = output_tmpfile.name
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Load model
        model = main_model()

        st.success("Processing video")        
        # Process frames
        progress_bar = st.progress(0)
        frame_count = 0
        total_class_counts = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            results = model(frame, conf=confidence)
            processed_frame = results[0].plot()
            
            # Update class counts
            for box in results[0].boxes:
                class_name = model.names[int(box.cls)]
                total_class_counts[class_name] = total_class_counts.get(class_name, 0) + 1
            
            # Write processed frame
            out.write(processed_frame)
            
            # Update progress
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
        
        # Release resources
        cap.release()
        out.release()
        
        # Read the processed video into memory
        with open(output_path, 'rb') as f:
            processed_video_bytes = f.read()
            
        return processed_video_bytes, total_class_counts
        
    finally:
        # Clean up temporary files
        if os.path.exists(video_path):
            os.unlink(video_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

if source_type == "Image":
    # Image processing code
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', "bmp", "webp"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image",
                    use_column_width=True)
        else:
            st.info("Please upload an image.")
            
    with col2:
        if st.sidebar.button('Detect') and uploaded_file is not None:
            try:
                model = main_model()
                results = model(
                    source=uploaded_image,
                    conf=confidence,
                    device="cpu"
                )
                boxes = results[0].boxes
                res_plotted = results[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detect Image',
                         use_column_width=True)
                                
                # Count objects
                class_counts = {}
                for cls in boxes.cls:
                    class_name = model.names[int(cls)]
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
                
                # Display counts
                table_data = [{"Class": class_name, "Count": count} 
                             for class_name, count in class_counts.items()]
                st.write("Number of objects for each detected class:")
                st.table(table_data)
                
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                st.exception(ex)
        else:
            st.error("Please upload an image first!")

else:  # Video processing
    uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Display original video in a smaller size
        video_bytes = uploaded_file.read()
        
        # Create a container with custom width
        video_container = st.container()
        with video_container:
            col1, col2, col3 = st.columns([1,2,1])  # Creates three columns with middle one being larger
            with col2:  # Use the middle column for the video
                st.video(video_bytes)
        
        if st.sidebar.button('Detect'):
            try:
                # Process video
                processed_video_bytes, total_class_counts = process_uploaded_video(video_bytes)
                
                # Display detection statistics
                st.write("Total detections throughout the video:")
                table_data = [{"Class": class_name, "Total Count": count} 
                             for class_name, count in total_class_counts.items()]
                st.table(table_data)
                
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.download_button(
                        label="⬇️ Download processed video",
                        data=processed_video_bytes,
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )
                
            except Exception as ex:
                st.exception(ex)
                st.error("An error occurred during video processing. Please try again.")
    else:
        st.info("Please upload a video.")


