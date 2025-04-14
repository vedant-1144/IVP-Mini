import streamlit as st
import cv2
import numpy as np
import os
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
        original_size = get_file_size(original_image)
        
        # Create tabs for different processes
        tab1, tab2, tab3 = st.tabs(["📸 Enhancement", "📊 Analysis", "📑 Export"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Document")
                st.image(original_image, use_container_width=True)
                
                # Blur detection
                is_blurry, blur_score = check_blur(original_image)
                if is_blurry:
                    st.markdown('<div class="warning">⚠️ Warning: Image appears to be blurry</div>', 
                              unsafe_allow_html=True)
                
                # Enhancement controls
                st.subheader("Enhancement Settings")
                contrast = st.slider("Contrast", 1.0, 3.0, 1.5, 0.1)
                brightness = st.slider("Brightness", 0, 100, 20)
            
            with col2:
                st.subheader("Enhanced Document")
                enhanced, is_blurry, _ = enhance_document(original_image)
                st.image(enhanced, use_container_width=True)
                
                # Size comparison
                enhanced_size = get_file_size(enhanced)
                st.markdown(f"""
                <div class="stats">
                    📊 Size Comparison:<br>
                    Original: {original_size:.1f}KB<br>
                    Enhanced: {enhanced_size:.1f}KB<br>
                    Reduction: {((original_size-enhanced_size)/original_size*100):.1f}%
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Original Histogram")
                plot_histogram(original_image, "Original")
                st.image("output/Original.png", use_container_width=True)
                
                st.subheader("Edge Detection")
                edges = detect_edges(enhanced)
                st.image(edges, use_container_width=True)
            
            with col4:
                st.subheader("Enhanced Histogram")
                plot_histogram(enhanced, "Enhanced")
                st.image("output/Enhanced.png", use_container_width=True)
                
                st.subheader("Auto-Cropped")
                cropped = crop_document(enhanced, edges)
                st.image(cropped, use_container_width=True)
        
        with tab3:
            st.subheader("Export Options")
            
            col5, col6 = st.columns(2)
            with col5:
                watermark = st.text_input("Add Watermark (optional)", "")
                if watermark:
                    enhanced = add_watermark(enhanced, watermark)
                
                _, buffer = cv2.imencode('.png', enhanced)
                st.download_button(
                    "📥 Download Enhanced Image",
                    buffer.tobytes(),
                    "enhanced_document.png",
                    "image/png",
                    use_container_width=True
                )
            
            with col6:
                if st.button("Export as PDF", use_container_width=True):
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
    if st.button("Start Camera"):
        cap = cv2.VideoCapture(0)
        captured_images = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display frame
            st.image(frame, channels="BGR", use_container_width=True)
            
            # Auto-capture when document is detected
            edges = detect_edges(frame)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 50000:  # Min area threshold
                    enhanced, _, _ = enhance_document(frame)
                    captured_images.append(enhanced)
                    st.success(f"Page {len(captured_images)} captured!")
                    
            if st.button("Finish Scanning"):
                break
        
        cap.release()
        
        if captured_images:
            # Save all pages to PDF
            temp_paths = []
            for i, img in enumerate(captured_images):
                path = f"output/page_{i}.png"
                cv2.imwrite(path, img)
                temp_paths.append(path)
            
            save_to_pdf(temp_paths, "scanned_document.pdf")
            with open("scanned_document.pdf", "rb") as f:
                st.download_button(
                    "📥 Download Multi-Page PDF",
                    f.read(),
                    "scanned_document.pdf",
                    "application/pdf",
                    use_container_width=True
                )