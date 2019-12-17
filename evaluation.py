from sklearn import metrics

from typing import List


def print_metrics(predicted_labels: List[int], ground_truth_labels: List[int]):
    print(f"Predicted:    {predicted_labels}")
    print(f"Ground Truth: {ground_truth_labels}")

    ari = metrics.adjusted_rand_score(ground_truth_labels, predicted_labels)
    ami = metrics.adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
    fmi = metrics.fowlkes_mallows_score(ground_truth_labels, predicted_labels)

    print(f"Adjusted Rand Index (ARI):         {ari}")
    print(f"Adjusted Mutual Information (AMI): {ami}")
    print(f"Fowlkes-Mallows index (FMI):       {fmi}")
