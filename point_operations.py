import cv2
import numpy as np

def image_negative(image):
    return 255 - image

def threshold_image(image, threshold=127, inverse=False):
    if inverse:
        return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]

def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.multiply(hsv[:,:,2], factor)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def bit_plane_slice(image, bit=7):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.uint8(((grayscale >> bit) & 1) * 255)