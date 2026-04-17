import cv2
import numpy as np

def stability_score(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,50,150)

    projection = np.sum(edges,axis=1)

    variation = np.var(projection)

    score = min(100, variation/100)

    return score