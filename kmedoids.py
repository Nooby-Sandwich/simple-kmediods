import numpy as np
import random

def k_medoids(X, k, max_iter=300):
    # Randomly initialize medoids
    medoids = X[random.sample(range(len(X)), k)]
    
    for _ in range(max_iter):
        # Assign each point to the nearest medoid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - medoids, axis=2), axis=1)

        # Update medoids
        new_medoids = np.array([X[labels == i][np.argmin(np.sum(np.linalg.norm(X[labels == i] - point, axis=1)))] for i, point in enumerate(medoids)])
        
        # Stop if medoids don't change
        if np.all(medoids == new_medoids):
            break
        medoids = new_medoids

    return medoids, labels
