import numpy as np
from pyod.models.iforest import IForest

def detect_anomalies(values, contamination=0.05):
    # Reshape for PyOD (expects 2D array)
    X = np.array(values).reshape(-1, 1)

    model = IForest(contamination=contamination, random_state=42)
    model.fit(X)

    # Labels: 0 = normal, 1 = anomaly
    labels = model.labels_
    scores = model.decision_scores_

    return labels, scores
