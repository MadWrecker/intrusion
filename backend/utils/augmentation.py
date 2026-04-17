import cv2
import numpy as np
import random

def _rotate_image(image, angle):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=(0,0,0))

def _shear_x(image, angle):
    # Simulates YAW (left/right turn)
    h, w = image.shape[:2]
    shear = np.tan(np.radians(angle))
    M = np.float32([[1, shear, max(0, -shear*h/2)], [0, 1, 0]])
    return cv2.warpAffine(image, M, (w, h), borderValue=(0,0,0))

def _shear_y(image, angle):
    # Simulates PITCH (up/down tilt)
    h, w = image.shape[:2]
    shear = np.tan(np.radians(angle))
    M = np.float32([[1, 0, 0], [shear, 1, max(0, -shear*w/2)]])
    return cv2.warpAffine(image, M, (w, h), borderValue=(0,0,0))

def _change_brightness_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def _blur(image, ksize=3):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def _sharpen(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def _zoom(image, factor=1.1):
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(image, (new_w, new_h))
    
    if factor > 1:
        # Crop center
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        return resized[start_y:start_y+h, start_x:start_x+w]
    else:
        # Pad
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        canvas = np.zeros_like(image)
        end_y = min(pad_y+new_h, h)
        end_x = min(pad_x+new_w, w)
        vis_h = end_y - pad_y
        vis_w = end_x - pad_x
        canvas[pad_y:end_y, pad_x:end_x] = resized[:vis_h, :vis_w]
        return canvas

def generate_augmentations(base_img_rgb):
    """
    Transform a single uploaded face image into a strong, multi-angle identity representation.
    """
    augmentations = [base_img_rgb] # 0 deg (front)
    
    # 1. Base Geometries
    # Yaw
    yaws = []
    for angle in [-20, -10, 10, 20]:
        img = _shear_x(base_img_rgb, angle)
        yaws.append(img)
        augmentations.append(img)
        
    # Roll
    rolls = []
    for angle in [-10, 10]:
        img = _rotate_image(base_img_rgb, angle)
        rolls.append(img)
        augmentations.append(img)
        
    # Pitch
    pitches = []
    for angle in [-10, 10]:
        img = _shear_y(base_img_rgb, angle)
        pitches.append(img)
        augmentations.append(img)
        
    # 3. Return only a small subset of core geometric variations to keep processing extremely fast
    # We prioritize speed and rely on the base image being clear.
    fast_augmentations = [base_img_rgb]
    if yaws: fast_augmentations.append(yaws[0])
    if yaws and len(yaws) > 1: fast_augmentations.append(yaws[1])
    if rolls: fast_augmentations.append(rolls[0])
    if pitches: fast_augmentations.append(pitches[0])
    
    return fast_augmentations
