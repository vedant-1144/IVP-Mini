import cv2
import numpy as np

def minimum_filter(image, kernel_size=3):
    return cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8))

def maximum_filter(image, kernel_size=3):
    return cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))

def average_filter(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)

def gaussian_filter(image, kernel_size=3, sigma=1):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)