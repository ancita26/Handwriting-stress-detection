import cv2
import numpy as np

def extract_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Edge detection
    edges = cv2.Canny(blur,50,150)

    # Binary image
    thresh = cv2.threshold(gray,0,255,
            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    # Contours for handwriting
    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    areas = []
    widths = []
    heights = []

    for c in contours:

        area = cv2.contourArea(c)

        if area > 50:

            x,y,w,h = cv2.boundingRect(c)

            areas.append(area)
            widths.append(w)
            heights.append(h)

    if len(areas)==0:

        areas=[0]
        widths=[0]
        heights=[0]

    # Feature calculations
    baseline_angle = np.mean(edges)
    letter_size = np.mean(heights)
    word_spacing = np.std(widths)
    slant_angle = np.var(edges)
    writing_pressure = np.mean(gray)
    stroke_density = len(contours)
    pen_lift = np.std(areas)
    writing_speed = np.mean(widths)

    features = [
        baseline_angle,
        letter_size,
        word_spacing,
        slant_angle,
        writing_pressure,
        stroke_density,
        pen_lift,
        writing_speed
    ]

    return features