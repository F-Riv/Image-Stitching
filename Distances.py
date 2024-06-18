import numpy as np





# Normalize the descriptor vector
def normalize_descriptor(descriptor):
    norm = np.linalg.norm(descriptor)
    if norm == 0:
        return descriptor
    return descriptor / norm

# Normalized correlation between descriptors 
def normalized_corr(des_1, des_2): 
    score= np.dot(normalize_descriptor(des_1), normalize_descriptor(des_2).T)
    return np.mean(score)


# Compute Euclidean distance between two descriptors
def euclidean_distance(des_1, des_2):
    return np.linalg.norm(des_1 - des_2)


# Compute distances using Euclidean distance after normalization and normalized correlation 
def compute_distance(method, des_1, des_2):
    distances = np.zeros((len(des_1), len(des_2)))
    for i, desc1 in enumerate(des_1):
        for j, desc2 in enumerate(des_2):
            if method == "eucl":
                distances[i, j] = euclidean_distance(normalize_descriptor(desc1), normalize_descriptor(desc2))
                #print(distances_euclidean)
            elif method == "correlation":
                distances[i, j] = normalized_corr(distances.normalize_descriptor(desc1), normalize_descriptor(desc2))
            else: print("error")    
    return distances


# Select n top matches
def select_best_matches(distances, top_n):

    sorted_indices = np.argsort(distances, axis=None)[:top_n]
    top_matches = np.unravel_index(sorted_indices, distances.shape)
    top_matches = list(zip(*top_matches)) 
    print(f"For top_n {top_n}, number of matches: {len(top_matches)} ")
    return top_matches    