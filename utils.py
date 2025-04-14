import matplotlib.pyplot as plt
from fpdf import FPDF
import cv2
import numpy as np

def plot_histogram(img, title='Histogram'):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure()
    plt.title(title)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"output/{title}.png")
    plt.close()

def save_to_pdf(image_paths, filename='scanned_doc.pdf'):
    pdf = FPDF()
    for path in image_paths:
        pdf.add_page()
        pdf.image(path, x=10, y=10, w=190)
    pdf.output(filename)

def add_watermark(image, text):
    """Add watermark to the bottom of the image"""
    height, width = image.shape[:2]
    # Create a copy of the image
    watermarked = image.copy()
    
    # Set the font and text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    color = (128, 128, 128)  # Gray color
    
    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # Calculate text position (bottom-right corner with padding)
    padding = 10
    text_x = width - text_size[0] - padding
    text_y = height - padding
    
    # Add semi-transparent background for better visibility
    overlay = watermarked.copy()
    bg_rect_pts = np.array([[text_x - 5, text_y + 5], 
                           [text_x + text_size[0] + 5, text_y + 5],
                           [text_x + text_size[0] + 5, text_y - text_size[1] - 5],
                           [text_x - 5, text_y - text_size[1] - 5]])
    cv2.fillPoly(overlay, [bg_rect_pts], (255, 255, 255))
    cv2.addWeighted(overlay, 0.3, watermarked, 0.7, 0, watermarked)
    
    # Add text
    cv2.putText(watermarked, text, (text_x, text_y), 
                font, font_scale, color, font_thickness)
    
    return watermarked