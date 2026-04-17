import cv2
import numpy as np
from models.recognizer import FastFaceRecognizer

import sys
sys.stdout = open("arcface_debug.txt", "w")

def test_embeddings():
    recognizer = FastFaceRecognizer()
    
    # Create two random distinct images
    img1 = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    
    # We bypass landmarks to just force the model to infer
    # In recognizer.py, if landmarks=None, it returns None. Wait!
    # Let's mock landmarks for a valid alignment
    lms1 = [[38, 51], [73, 51], [56, 71], [41, 92], [70, 92]]
    
    emb1 = recognizer.get_embedding(img1, lms1)
    emb2 = recognizer.get_embedding(img2, lms1)
    
    if emb1 is None or emb2 is None:
        print("Failed to get embeddings.")
        return
        
    dot = np.dot(emb1, emb2)
    print(f"Similarity between two RANDOM noise images: {dot:.4f}")
    print(f"Random 1 Emb[:5]: {emb1[:5]}")
    print(f"Random 2 Emb[:5]: {emb2[:5]}")
    
    # Let's test two totally different real colors
    img3 = np.zeros((112, 112, 3), dtype=np.uint8)
    img4 = np.ones((112, 112, 3), dtype=np.uint8) * 255
    emb3 = recognizer.get_embedding(img3, lms1)
    emb4 = recognizer.get_embedding(img4, lms1)
    dot2 = np.dot(emb3, emb4)
    print(f"Similarity between BLACK and WHITE images: {dot2:.4f}")
    print(f"Black Emb[:5]: {emb3[:5]}")
    print(f"White Emb[:5]: {emb4[:5]}")

if __name__ == '__main__':
    test_embeddings()
