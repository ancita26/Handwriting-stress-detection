Handwriting-Based Mental Health Monitoring System

📌 Project Description

This project is a machine learning-based system that analyzes handwriting patterns to detect mental health conditions such as stress and emotional state. The system accepts handwriting input through image upload or mobile camera and provides real-time predictions along with visualization outputs like heatmap and graphs.

---

🎯 Objectives

- To detect stress level using handwriting features
- To classify emotional state (Calm / Anxiety / Anger)
- To provide real-time analysis using machine learning
- To visualize results using graphs and heatmaps

---

🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- Scikit-learn
- NumPy & Pandas
- Matplotlib

---

🧠 Algorithms Used

- Random Forest (for stress prediction)
- Image Processing (OpenCV)
- Feature Extraction Techniques

---

📂 Project Structure

model/
   trained_model.pkl

history/
   stress_history.csv

utils/
   image_processing.py
   personality_detector.py
   writer_identifier.py
   stress_meter.py
   stress_heatmap.py
   stress_highlight.py
   history_graph.py
   writing_stability.py

app.py
predict.py
train_model.py

---

▶️ How to Run the Project

1. Install required libraries:
   pip install -r requirements.txt

2. Run the application:
   streamlit run app.py

3. Open in browser and upload handwriting image or use camera

---

📊 Output

- Stress Score (0–100)
- Stress Level (Low / Medium / High)
- Emotion Detection
- Heatmap Visualization
- History Graph

---

📈 Performance

- Model Accuracy: ~75%
- Real-time prediction supported

---

🔮 Future Enhancements

- Improve dataset for better accuracy
- Convert into mobile application
- Add more psychological analysis features

---

👩‍💻 Author

- ZENO P
- MCA Project

---

📚 Conclusion

This project demonstrates how handwriting analysis combined with machine learning can be used for early detection of mental health conditions in a simple and non-invasive way.