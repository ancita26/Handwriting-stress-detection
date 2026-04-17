def stress_level(score):

    if score < 40:
        return "Normal (No Stress)"

    elif score < 70:
        return "Moderate Stress"

    else:
        return "High Stress"