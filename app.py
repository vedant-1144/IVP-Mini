import streamlit as st
import cv2
import numpy as np
import os
import time
from datetime import datetime
from processing import *
from utils import plot_histogram, save_to_pdf, add_watermark

# Page setup and styling
st.set_page_config(page_title="AI Document Scanner", layout="wide")

# Enhanced CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTitle { text-align: center; color: #1E3D59; }
    .warning { 
        padding: 1rem; 
        background-color: #fff3cd; 
        border-left: 5px solid #ffeeba;
        margin: 1rem 0;
    }
    .success {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #c3e6cb;
        margin: 1rem 0;
    }
    .stats {
        padding: 1rem;
        background-color: #e9ecef;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Advanced Document Scanner & Enhancer")

# Mode selection
mode = st.radio("Select Input Mode", ["Single Image", "Multi-Page Scan"])

if mode == "Single Image":
    uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)
        
        # Create tabs for different processes
        tab1, tab2, tab3 = st.tabs(["📸 Basic Processing", "📊 Advanced Processing", "📑 Export"])
        
        with tab1:
            st.subheader("1️⃣ Original Image")
            st.image(original_image, channels="BGR", use_container_width=True)
            
            # Basic Enhancement Controls
            st.subheader("2️⃣ Basic Enhancement")
            col1, col2 = st.columns(2)
            with col1:
                contrast = st.slider("Contrast", 1.0, 3.0, 1.5, 0.1)
                brightness = st.slider("Brightness", 0, 100, 20)
                
                # Apply basic enhancement
                enhanced = cv2.convertScaleAbs(original_image, alpha=contrast, beta=brightness)
                st.image(enhanced, channels="BGR", use_container_width=True)
            
            with col2:
                # Show immediate size comparison
                original_size = get_file_size(original_image)
                enhanced_size = get_file_size(enhanced)
                st.markdown(f"""
                <div class="stats">
                    📊 Size Comparison:
                    - Original: {original_size:.1f}KB
                    - Enhanced: {enhanced_size:.1f}KB
                    - Reduction: {((original_size-enhanced_size)/original_size*100):.1f}%
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Edge Detection
                st.subheader("1️⃣ Edge Detection")
                edges = detect_edges(enhanced)
                st.image(edges, use_container_width=True)
                
                # Blur Detection
                st.subheader("2️⃣ Blur Analysis")
                is_blurry, blur_score = check_blur(original_image)
                st.markdown(f"""
                <div class="{'warning' if is_blurry else 'success'}">
                    {'⚠️' if is_blurry else '✅'} Blur Score: {blur_score:.2f}<br>
                    Status: {'Blurry' if is_blurry else 'Sharp'} Image
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Histogram Analysis
                st.subheader("3️⃣ Histogram Analysis")
                plot_histogram(enhanced, "Enhanced")
                st.image("output/Enhanced.png", use_container_width=True)
                
                # Auto-Cropping
                st.subheader("4️⃣ Auto-Cropped Result")
                cropped = crop_document(enhanced, edges)
                if cropped is not None:
                    st.image(cropped, channels="BGR", use_container_width=True)
                else:
                    st.warning("Could not detect document borders for cropping")
        
        with tab3:
            st.subheader("Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # Watermark Option
                watermark = st.text_input("Add Watermark Text (optional)", "")
                if watermark:
                    enhanced = add_watermark(enhanced, watermark)
                    st.image(enhanced, channels="BGR", use_container_width=True)
                
                # Download Enhanced Image
                _, buffer = cv2.imencode('.png', enhanced)
                st.download_button(
                    "📥 Download Enhanced Image",
                    buffer.tobytes(),
                    "enhanced_document.png",
                    "image/png",
                    use_container_width=True
                )
            
            with col2:
                # PDF Export with processed image
                if st.button("Generate PDF", use_container_width=True):
                    temp_path = "output/enhanced.png"
                    cv2.imwrite(temp_path, enhanced)
                    save_to_pdf([temp_path], "enhanced_document.pdf")
                    with open("enhanced_document.pdf", "rb") as f:
                        st.download_button(
                            "📥 Download PDF",
                            f.read(),
                            "enhanced_document.pdf",
                            "application/pdf",
                            use_container_width=True
                        )

else:  # Multi-Page Scan mode
    st.info("📸 Multi-Page Scan Mode")
    col1, col2 = st.columns(2)
    
    with col1:
        recording_duration = st.slider("Recording Duration (seconds)", 5, 30, 10)
    
    start_recording = st.button("Start Recording", key="start_record")
    
    if start_recording:
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        frames = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Record video
        start_time = time.time()
        while (time.time() - start_time) < recording_duration:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                # Update progress
                progress = (time.time() - start_time) / recording_duration
                progress_bar.progress(progress)
                status_text.text(f"Recording: {int((time.time() - start_time))} seconds")
                
                # Show live preview
                preview = st.empty()
                preview.image(frame, channels="BGR", use_container_width=True)
        
        cap.release()
        st.success("Recording Complete! Processing frames...")
        
        # Process frames to find best quality shots
        good_frames = []
        for frame in frames:
            # Check blur and document presence
            is_blurry, blur_score = check_blur(frame)
            edges = detect_edges(frame)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0 and not is_blurry:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 50000:  # Min area threshold
                    enhanced, _, _ = enhance_document(frame)
                    good_frames.append(enhanced)
        
        # Show results
        if good_frames:
            st.success(f"Found {len(good_frames)} good quality document frames!")
            
            # Display and allow selection of frames
            selected_frames = []
            for i, frame in enumerate(good_frames):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
                with col2:
                    if st.checkbox(f"Select Frame {i+1}", value=True):
                        selected_frames.append(frame)
            
            if selected_frames and st.button("Create PDF", key="create_pdf"):
                # Save selected frames
                temp_paths = []
                for i, frame in enumerate(selected_frames):
                    path = f"output/page_{i}.png"
                    cv2.imwrite(path, frame)
                    temp_paths.append(path)
                
                # Generate PDF
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"scanned_document_{timestamp}.pdf"
                save_to_pdf(temp_paths, pdf_filename)
                
                # Offer download
                with open(pdf_filename, "rb") as f:
                    st.download_button(
                        "📥 Download PDF",
                        f.read(),
                        pdf_filename,
                        "application/pdf",
                        use_container_width=True,
                        key="download_pdf"
                    )
        else:
            st.warning("No good quality document frames found. Please try recording again.")