def detect_personality(features, stress_score, stability):

    baseline_angle = features[0]
    letter_size = features[1]
    spacing = features[2]
    slant = features[3]

    # confident writing
    if stress_score < 30 and stability > 25:
        return "Confidence"

    # aggressive / stressed writing
    elif stress_score > 70:
        return "Aggression"

    # nervous writing
    elif spacing > 6 and slant > 10:
        return "Nervousness"

    # balanced personality
    else:
        return "Calm Personality"