import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def show_history_graph():

    try:
        data = pd.read_csv("history/stress_history.csv")

        if len(data) == 0:
            st.warning("No stress history available yet.")
            return

        stress_scores = data["stress_score"]

        fig, ax = plt.subplots()

        ax.plot(
            stress_scores,
            marker="o",
            linewidth=3
        )

        ax.set_title("Handwriting Stress Trend", fontsize=16)

        ax.set_xlabel("Session Number", fontsize=12)

        ax.set_ylabel("Stress Score", fontsize=12)

        ax.grid(True)

        for i, value in enumerate(stress_scores):
            ax.text(i, value+1, str(round(value,1)), ha='center')

        st.pyplot(fig)

    except Exception as e:

        st.error("Unable to load stress history graph.")