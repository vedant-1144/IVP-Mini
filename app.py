import cv2
import numpy as np
import streamlit as st
import os
import time
from datetime import datetime
from processing import *
from utils import *
from point_operations import *
from mask_operations import *

# Add this helper function at the top with other imports
def add_image_download_button(image, filename, label="Download Image"):
    """Helper function to create download button for an image"""
    success, buffer = cv2.imencode('.png', image)
    if success:
        btn = st.download_button(
            label=f"📥 {label}",
            data=buffer.tobytes(),
            file_name=filename,
            mime="image/png",
            use_container_width=True
        )
    return success

# Page setup and styling
st.set_page_config(page_title="Advanced Document Scanner & Enhancer", layout="wide")

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
            # Create subtabs for different processing techniques
            process_tab1, process_tab2, process_tab3 = st.tabs(["Point & Mask Processing", "Edge Detection", "Advanced"])
            
            with process_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    # Point Processing
                    st.subheader("1️⃣ Point Processing")
                    point_operation = st.selectbox(
                        "Select Operation",
                        ["Image Negative", "Thresholding", "Brightness Adjustment", "Bit-Plane Slicing"]
                    )
                    
                    if point_operation == "Image Negative":
                        processed = image_negative(enhanced)
                        st.image(processed, channels="BGR", use_container_width=True)
                        add_image_download_button(processed, "negative_image.png", "Download Negative Image")
                        
                    elif point_operation == "Thresholding":
                        threshold = st.slider("Threshold Value", 0, 255, 127)
                        inverse = st.checkbox("Inverse Threshold")
                        processed = threshold_image(enhanced, threshold, inverse)
                        st.image(processed, channels="BGR", use_container_width=True)
                        add_image_download_button(processed, "threshold_image.png", "Download Threshold Image")
                        
                    elif point_operation == "Brightness Adjustment":
                        factor = st.slider("Brightness Factor", 0.1, 3.0, 1.0, 0.1)
                        processed = adjust_brightness(enhanced, factor)
                        st.image(processed, channels="BGR", use_container_width=True)
                        add_image_download_button(processed, "brightness_adjusted.png", "Download Adjusted Image")
                        
                    else:  # Bit-Plane Slicing
                        bit = st.slider("Bit Plane (0-7)", 0, 7, 7)
                        processed = bit_plane_slice(enhanced, bit)
                        st.image(processed, use_container_width=True)
                        add_image_download_button(processed, f"bitplane_{bit}.png", f"Download Bit-Plane {bit}")
                
                with col2:
                    # Mask Processing
                    st.subheader("2️⃣ Mask Processing")
                    mask_type = st.selectbox(
                        "Filter Type",
                        ["Minimum", "Maximum", "Average", "Gaussian", "Median"]
                    )
                    
                    kernel_size = st.slider("Kernel Size", 3, 9, 3, 2)
                    
                    if mask_type == "Minimum":
                        processed = minimum_filter(enhanced, kernel_size)
                    elif mask_type == "Maximum":
                        processed = maximum_filter(enhanced, kernel_size)
                    elif mask_type == "Average":
                        processed = average_filter(enhanced, kernel_size)
                    elif mask_type == "Gaussian":
                        sigma = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
                        processed = gaussian_filter(enhanced, kernel_size, sigma)
                    else:  # Median
                        processed = median_filter(enhanced, kernel_size)
                    
                    st.image(processed, channels="BGR", use_container_width=True)
                    add_image_download_button(processed, f"{mask_type.lower()}_filter.png", f"Download {mask_type} Filtered")
            
            with process_tab2:
                col1, col2 = st.columns(2)
                with col1:
                    # Edge Detection
                    st.subheader("3️⃣ Edge Detection")
                    edge_type = st.selectbox(
                        "Edge Detection Method",
                        ["Sobel", "Prewitt", "Roberts", "Canny"]
                    )
                    
                    if edge_type == "Sobel":
                        dx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
                        dy = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
                        edges = np.sqrt(dx**2 + dy**2).astype(np.uint8)
                    elif edge_type == "Prewitt":
                        kernelx = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
                        kernely = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])
                        dx = cv2.filter2D(enhanced, -1, kernelx)
                        dy = cv2.filter2D(enhanced, -1, kernely)
                        edges = np.sqrt(dx**2 + dy**2).astype(np.uint8)
                    elif edge_type == "Roberts":
                        kernelx = np.array([[1,0], [0,-1]])
                        kernely = np.array([[0,1], [-1,0]])
                        dx = cv2.filter2D(enhanced, -1, kernelx)
                        dy = cv2.filter2D(enhanced, -1, kernely)
                        edges = np.sqrt(dx**2 + dy**2).astype(np.uint8)
                    else:  # Canny
                        threshold1 = st.slider("Threshold 1", 0, 255, 100)
                        threshold2 = st.slider("Threshold 2", 0, 255, 200)
                        edges = cv2.Canny(enhanced, threshold1, threshold2)
                    
                    st.image(edges, use_container_width=True)
                    add_image_download_button(edges, f"{edge_type.lower()}_edges.png", f"Download {edge_type} Edges")
                
                with col2:
                    # Histogram Equalization
                    st.subheader("4️⃣ Histogram Equalization")
                    eq_type = st.selectbox(
                        "Equalization Method",
                        ["Global", "Adaptive", "CLAHE"]
                    )
                    
                    if eq_type == "Global":
                        img_yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
                        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                        hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                    elif eq_type == "Adaptive":
                        hist_eq = cv2.equalizeHist(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
                    else:  # CLAHE
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        img_yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
                        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
                        hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                    
                    st.image(hist_eq, channels="BGR", use_container_width=True)
                    add_image_download_button(hist_eq, f"{eq_type.lower()}_equalized.png", f"Download {eq_type} Equalized")
            
            with process_tab3:
                # DCT Compression
                st.subheader("5️⃣ DCT Compression")
                quality = st.slider("Compression Quality (lower = more compression)", 1, 100, 50)
                
                def dct_compress(img, quality_percent):
                    # Convert to YCrCb
                    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
                    channels = cv2.split(img_yuv)
                    compressed_channels = []
                    
                    for channel in channels:
                        # Apply DCT
                        dct = cv2.dct(np.float32(channel))
                        # Zero out high-frequency components based on quality
                        thresh = np.percentile(np.abs(dct), 100 - quality_percent)
                        dct[np.abs(dct) < thresh] = 0
                        # Inverse DCT
                        compressed = cv2.idct(dct)
                        compressed_channels.append(np.uint8(compressed))
                    
                    # Merge channels and convert back to BGR
                    compressed_yuv = cv2.merge(compressed_channels)
                    return cv2.cvtColor(compressed_yuv, cv2.COLOR_YCR_CB2BGR)
                
                compressed = dct_compress(enhanced, quality)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original Enhanced Image")
                    st.image(enhanced, channels="BGR", use_container_width=True)
                with col2:
                    st.write("DCT Compressed Image")
                    st.image(compressed, channels="BGR", use_container_width=True)
                    
                    # Show compression stats
                    original_size = get_file_size(enhanced)
                    compressed_size = get_file_size(compressed)
                    st.markdown(f"""
                    <div class="stats">
                        📊 Compression Stats:
                        - Original: {original_size:.1f}KB
                        - Compressed: {compressed_size:.1f}KB
                        - Reduction: {((original_size-compressed_size)/original_size*100):.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                # RLE Compression
                st.subheader("5️⃣ RLE Compression")
                
                # Convert to grayscale for better compression visualization
                gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                
                # Apply RLE compression
                encoded_data, original_shape = rle_encode(enhanced)
                compressed_img = rle_decode(encoded_data, original_shape)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original Enhanced Image")
                    st.image(enhanced, channels="BGR", use_container_width=True)
                    
                    # Show original histogram
                    st.write("Original Histogram")
                    plot_histogram(enhanced, "Original")
                    st.image("output/Original.png", use_container_width=True)
                
                with col2:
                    st.write("RLE Compressed Image")
                    st.image(compressed_img, channels="BGR", use_container_width=True)
                    add_image_download_button(compressed_img, "rle_compressed.png", "Download RLE Compressed")
                    
                    # Show compressed histogram
                    st.write("Compressed Histogram")
                    plot_histogram(compressed_img, "Compressed")
                    st.image("output/Compressed.png", use_container_width=True)
                    
                    # Show compression stats
                    original_size = get_file_size(enhanced)
                    compressed_size = len(encoded_data) / 1024  # Convert to KB
                    compression_ratio = (original_size - compressed_size) / original_size * 100
                    
                    st.markdown(f"""
                    <div class="stats">
                        📊 RLE Compression Stats:
                        - Original Size: {original_size:.1f}KB
                        - Compressed Size: {compressed_size:.1f}KB
                        - Compression Ratio: {compression_ratio:.1f}%
                        - Encoded Data Length: {len(encoded_data)} bytes
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add option to download compressed data
                if st.button("Save Compressed Data", use_container_width=True):
                    # Save encoded data to file
                    np.save("output/compressed_data.npy", encoded_data)
                    with open("output/compressed_data.npy", "rb") as f:
                        st.download_button(
                            "📥 Download Compressed Data",
                            f.read(),
                            "compressed_data.npy",
                            "application/octet-stream",
                            use_container_width=True
                        )
        
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