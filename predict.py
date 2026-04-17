import joblib
import numpy as np
import cv2

model = joblib.load("model/random_forest_model.pkl")

def predict_stress(features, image=None):

    features = np.array(features).reshape(1,-1)

    prediction = model.predict(features)[0]

    probs = model.predict_proba(features)

    # safe probability handling
    if probs.shape[1] == 1:
        probability = probs[0][0]
    else:
        probability = probs[0][1]

    stress_score = probability * 70

    extra_score = 0

    if image is not None:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray,50,150)

        edge_density = np.mean(edges)

        variance = np.var(gray)

        extra_score = (edge_density*0.1) + (variance*0.01)

    stress_score = stress_score + extra_score

    stress_score = min(100, round(stress_score,2))

    return prediction, stress_score