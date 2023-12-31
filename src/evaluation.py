import random
from typing import Union, List

import config
import treepredict


def train_test_split(dataset, test_size: Union[float, int], seed=None) -> (List, List):
    if seed:
        random.seed(seed)

    # If test size is a float, use it as a percentage of the total rows
    # Otherwise, use it directly as the number of rows in the test dataset
    n_rows = len(dataset)
    if float(test_size) != int(test_size):
        test_size = int(n_rows * test_size)  # We need an integer number of rows

    # From all the rows index, we get a sample which will be the test dataset
    choices = list(range(n_rows))
    test_rows = random.choices(choices, k=test_size)

    test = [row for (i, row) in enumerate(dataset) if i in test_rows]
    train = [row for (i, row) in enumerate(dataset) if i not in test_rows]

    return train, test


def get_accuracy(tree, dataset):
    correct_count = 0
    for row in dataset:
        result = treepredict.classify(tree, row[:-1])
        if result:
            result_value = list(result.keys())[0]
            if result_value == row[-1]:
                correct_count += 1
    return correct_count / len(dataset)


def mean(values: List[float]):
    return sum(values) / len(values)


def cross_validation(dataset, k, agg, seed, scoref, beta, threshold) -> float:
    random.seed(seed)
    random.shuffle(dataset)

    fold_size = len(dataset) // k
    folds = []
    for i in range(0, len(dataset), fold_size):
        folds.append(dataset[i:i + fold_size])

    accuracies = []
    for i in range(k):
        train = []
        test = []
        for j in range(k):
            if i == j:
                test = folds[j]
            else:
                train += folds[j]

        tree = treepredict.buildtree(train, scoref=scoref, beta=beta)
        accuracy = get_accuracy(tree, test)
        accuracies.append(accuracy)

    return agg(accuracies)


def find_best_threshold(dataset, thresholds, k, scoref, seed) -> (float, float):
    train, test = train_test_split(dataset, test_size=0.2, seed=seed)
    best_threshold = None
    best_accuracy = 0.0

    for threshold in thresholds:
        accuracy = cross_validation(train, k=k, agg=mean, seed=seed, scoref=scoref, beta=threshold, threshold=threshold)
        print(f"Threshold: {threshold}, Cross-Validation Accuracy: {round(accuracy, config.ROUND_DIGITS)}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    if best_threshold is None:
        print("There is no best threshold")
        return None, None

    tree = treepredict.buildtree(train, scoref=scoref, beta=best_threshold)
    test_accuracy = get_accuracy(tree, test)
    print(f"Test set accuracy with best threshold: {round(test_accuracy, config.ROUND_DIGITS)}")

    return best_threshold, best_accuracy


def test_find_best_threshold(filename) -> None:
    headers, data = treepredict.read(filename)
    best_threshold, best_accuracy = find_best_threshold(data, config.evaluation_thresholds, k=config.k,
                                                        scoref=treepredict.entropy, seed=config.seed)
    print(f"RESULT -> Best threshold: {best_threshold} with an accuracy of {round(best_accuracy, config.ROUND_DIGITS)}")


if __name__ == "__main__":
    # config.print_line(config.FILE1)  # decision_tree_example.txt
    # test_find_best_threshold(config.FILE1)

    config.print_line(config.FILE2)  # iris.csv
    test_find_best_threshold(config.FILE2)
