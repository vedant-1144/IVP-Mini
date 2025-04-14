import cv2
import numpy as np
from scipy.ndimage import variance

def check_blur(image):
    """Detect if image is blurry"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < 100, lap_var  # Returns (is_blurry, blur_score)

def enhance_contrast(img, alpha=1.5, beta=20):
    """Enhanced contrast with CLAHE"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

def sharpen_image(img):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def equalize_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 75, 200)

def crop_document(img, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return img[y:y+h, x:x+w]

def auto_rotate(image, edges):
    """Auto-rotate skewed document"""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def enhance_document(image):
    """Complete document enhancement pipeline"""
    # Check for blur
    is_blurry, blur_score = check_blur(image)
    
    # Basic enhancement
    enhanced = enhance_contrast(image)
    
    # Edge detection
    edges = detect_edges(enhanced)
    
    # Auto-rotate
    rotated = auto_rotate(enhanced, edges)
    
    # Auto-crop
    cropped = crop_document(rotated, detect_edges(rotated))
    
    # Final enhancement
    final = enhance_contrast(cropped)
    
    return final, is_blurry, blur_score

def dct_compress(img, quality=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_f = np.float32(gray) / 255.0
    dct = cv2.dct(img_f)
    dct[quality:, quality:] = 0
    idct = cv2.idct(dct)
    return np.uint8(idct * 255)

def get_file_size(image):
    """Get image file size in KB"""
    is_success, buffer = cv2.imencode(".jpg", image)
    if is_success:
        return len(buffer) / 1024
    return 0
