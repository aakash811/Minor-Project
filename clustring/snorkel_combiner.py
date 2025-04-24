# import numpy as np
# from collections import Counter

# def combine_cluster_labels(label_lists):
#     """
#     Combine multiple cluster label assignments (from different clustering algorithms)
#     using a simple voting mechanism per data point.
    
#     Args:
#         label_lists (list of arrays): Each array contains cluster labels for all points.
        
#     Returns:
#         combined_labels (np.array): Combined cluster labels.
#     """
#     label_lists = np.array(label_lists)  # shape: (num_algorithms, num_points)
#     num_points = label_lists.shape[1]
#     combined_labels = []

#     for i in range(num_points):
#         # Extract the labels for point i from all algorithms
#         point_labels = label_lists[:, i]
#         # Majority vote ignoring noise label (-1)
#         filtered_labels = [lbl for lbl in point_labels if lbl != -1]
#         if not filtered_labels:
#             combined_labels.append(-1)  # if all noise
#         else:
#             most_common_label = Counter(filtered_labels).most_common(1)[0][0]
#             combined_labels.append(most_common_label)
#     return np.array(combined_labels)

from snorkel.labeling.model import LabelModel
import numpy as np

def combine_cluster_labels(cluster_labels_list):
    """
    Combine multiple cluster labelings into one consensus labeling using Snorkel.
    Each cluster label array is treated as a labeling function output.
    """
    # Convert all cluster labels to int arrays and stack them: shape (num_methods, num_samples)
    label_matrix = np.vstack(cluster_labels_list)
    
    # Snorkel expects shape (num_samples, num_labeling_functions)
    label_matrix = label_matrix.T

    label_model = LabelModel(cardinality=np.max(label_matrix) + 1, verbose=True)
    label_model.fit(L_train=label_matrix, n_epochs=500, log_freq=100, seed=42)
    combined_labels = label_model.predict(label_matrix)
    return combined_labels
