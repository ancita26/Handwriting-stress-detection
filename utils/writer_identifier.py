import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

writers_database = []

def identify_writer(features):

    global writers_database

    features = np.array(features).reshape(1,-1)

    if len(writers_database) == 0:
        writers_database.append(features)
        return 1

    similarities = []

    for writer in writers_database:

        sim = cosine_similarity(features, writer)[0][0]

        similarities.append(sim)

    max_sim = max(similarities)

    if max_sim > 0.95:

        writer_id = similarities.index(max_sim) + 1

    else:

        writers_database.append(features)

        writer_id = len(writers_database)

    return writer_id