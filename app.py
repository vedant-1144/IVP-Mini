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

uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    
    # Create tabs for different processes
    tab1, tab2, tab3 = st.tabs(["📸 Basic Processing", "📊 Advanced Processing", "📑 Export"])
    
    with tab1:
        # Change layout to use columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1️⃣ Original Image")
            st.image(original_image, channels="BGR", use_container_width=True)
            
            # Show original size
            original_size = get_file_size(original_image)
            st.markdown(f"""
            <div class="stats">
                📊 Original Size: {original_size:.1f}KB
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("2️⃣ Basic Enhancement")
            # Controls for enhancement
            contrast = st.slider("Contrast", 1.0, 3.0, 1.5, 0.1)
            brightness = st.slider("Brightness", 0, 100, 20)
            
            # Apply basic enhancement
            enhanced = cv2.convertScaleAbs(original_image, alpha=contrast, beta=brightness)
            st.image(enhanced, channels="BGR", use_container_width=True)
            
            # Show enhancement stats
            enhanced_size = get_file_size(enhanced)
            st.markdown(f"""
            <div class="stats">
                📊 Enhancement Stats:
                - Enhanced Size: {enhanced_size:.1f}KB
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
                    # Convert to grayscale first
                    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                    kernelx = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])
                    kernely = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
                    dx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
                    dy = cv2.filter2D(gray, cv2.CV_64F, kernely)
                    # Combine and normalize
                    magnitude = np.sqrt(dx**2 + dy**2)
                    edges = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                elif edge_type == "Roberts":
                    # Convert to grayscale first
                    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                    kernelx = np.array([[1, 0], [0, -1]])
                    kernely = np.array([[0, 1], [-1, 0]])
                    dx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
                    dy = cv2.filter2D(gray, cv2.CV_64F, kernely)
                    # Combine and normalize
                    magnitude = np.sqrt(dx**2 + dy**2)
                    edges = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                else:  # Canny
                    threshold1 = st.slider("Threshold 1", 0, 255, 100)
                    threshold2 = st.slider("Threshold 2", 0, 255, 200)
                    edges = cv2.Canny(enhanced, threshold1, threshold2)

                # Add threshold control for Prewitt and Roberts
                if edge_type in ["Prewitt", "Roberts"]:
                    edge_threshold = st.slider("Edge Threshold", 0, 255, 30)
                    _, edges = cv2.threshold(edges, edge_threshold, 255, cv2.THRESH_BINARY)

                st.image(edges, use_container_width=True)
                add_image_download_button(edges, f"{edge_type.lower()}_edges.png", f"Download {edge_type} Edges")
            
            with col2:
                # Histogram Equalization
                st.subheader("4️⃣ Histogram Equalization")
                eq_type = "Global"
                
                img_yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                
                st.image(hist_eq, channels="BGR", use_container_width=True)
                add_image_download_button(hist_eq, f"{eq_type.lower()}_equalized.png", f"Download {eq_type} Equalized")
        
        with process_tab3:
            # Compression Section
            st.subheader("5️⃣ Image Compression")
            
            # Add compression level control
            compression_level = st.slider("Compression Level", 1, 8, 4)
            
            # Convert to grayscale and reduce color depth for compression
            gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            reduced_colors = (gray_enhanced // (2**compression_level)) * (2**compression_level)
            
            # Apply RLE compression
            encoded_data, original_shape = rle_encode(reduced_colors)
            compressed_img = rle_decode(encoded_data, original_shape)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original Enhanced Image")
                st.image(enhanced, channels="BGR", use_container_width=True)
                
                # Show original histogram
                st.write("Original Histogram")
                plot_histogram(enhanced, "Original")
                st.image("output/Original.png", use_container_width=True)
                
                # Original file size
                original_size = get_file_size(enhanced)
                st.markdown(f"Original Size: {original_size:.1f}KB")
            
            with col2:
                st.write("Compressed Image")
                st.image(compressed_img, channels="BGR", use_container_width=True)
                
                # Show compressed histogram
                st.write("Compressed Histogram")
                plot_histogram(compressed_img, "Compressed")
                st.image("output/Compressed.png", use_container_width=True)
                
                # Calculate compression statistics
                compressed_size = len(encoded_data) * 2 / 1024  # Convert bytes to KB
                compression_ratio = (original_size - compressed_size) / original_size * 100
                
                st.markdown(f"""
                <div class="stats">
                    📊 Compression Stats:
                    - Original Size: {original_size:.1f}KB
                    - Compressed Size: {compressed_size:.1f}KB
                    - Compression Ratio: {compression_ratio:.1f}%
                    - Color Levels: {256//(2**compression_level)}
                </div>
                """, unsafe_allow_html=True)
                
                add_image_download_button(compressed_img, 
                                        f"compressed_level_{compression_level}.png", 
                                        "Download Compressed Image")
            
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