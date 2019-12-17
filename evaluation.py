from sklearn import metrics

from typing import List

# left is 0, right is 1


def print_metrics(predicted_labels: List[int], ground_truth_labels: List[int]):
    print(f"Predicted:    {predicted_labels}")
    print(f"Ground Truth: {ground_truth_labels}")

    ari = metrics.adjusted_rand_score(ground_truth_labels, predicted_labels)
    ami = metrics.adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
    fmi = metrics.fowlkes_mallows_score(ground_truth_labels, predicted_labels)

    print(f"Adjusted Rand Index (ARI):         {ari}")
    print(f"Adjusted Mutual Information (AMI): {ami}")
    print(f"Fowlkes-Mallows index (FMI):       {fmi}")


def print_baseline_metrics():
    baseline_predictions = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1
    ]
    ground_truth = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]

    # subreddits = [
    #     'accidentallycommunist', 'againsthatesubreddits', 'anarchism', 'anarchocommunism', 'anarchy101',
    #     'ani_communism', 'antifascistsofreddit', 'antifastonetoss', 'antiwork', 'askaliberal',
    #     'bannedfromthe_donald', 'beto2020', 'bluemidterm2018', 'breadtube', 'centerleftpolitics',
    #     'chapotraphouse', 'chapotraphouse2', 'chomsky', 'circlebroke', 'askaconservative', 'benshapiro',
    #     'conservative', 'conservativelounge', 'conservatives', 'conservatives_only', 'cringeanarchy',
    #     'jordanpeterson', 'louderwithcrowder', 'metacanada', 'newpatriotism', 'paleoconservative',
    #     'republican', 'rightwinglgbt', 'shitpoliticssays', 'the_donald', 'thenewright', 'tuesday', 'walkaway'
    # ]
    score_correct = 0
    for i in range(len(baseline_predictions)):
        if baseline_predictions[i] == ground_truth[i]:
            score_correct += 1
    print("Baseline (on training set):")
    print_metrics(baseline_predictions, ground_truth)
    print("Baseline Accuracy (on training set): ", score_correct/len(ground_truth))

print_baseline_metrics()