import cv2
import numpy as np

def generate_heatmap(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,50,150)

    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image,0.6,heatmap,0.4,0)

    return overlay