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
    correct_counter = 0
    for row in dataset:
        result = treepredict.classify(tree, row[:-1])
        if result:
            result_value = list(result.keys())[0]
            if result_value == row[-1]:
                correct_counter += 1
    return correct_counter / len(dataset)


def mean(values: List[float]):
    return sum(values) / len(values)


def cross_validation(dataset, k, agg, scoref, beta, threshold) -> float:
    random.shuffle(dataset)

    fold_size = len(dataset) // k
    accuracies = []

    for i in range(k):
        start_test = i * fold_size
        end_test = start_test + fold_size
        test = dataset[start_test:end_test]
        train = dataset[:start_test] + dataset[end_test:]

        tree = treepredict.buildtree(train, scoref=scoref, beta=beta)
        treepredict.prune(tree, threshold)

        accuracy = get_accuracy(tree, test)
        accuracies.append(accuracy)

    return agg(accuracies)


def find_best_threshold(dataset) -> (float, float):
    train, test = train_test_split(dataset, test_size=config.test_size, seed=config.seed)
    best_threshold = None
    best_accuracy = 0.0

    thresholds = config.evaluation_thresholds
    k = config.k
    scoref = treepredict.entropy

    for threshold in thresholds:
        accuracy = cross_validation(train, k=k, agg=mean, scoref=scoref, beta=threshold, threshold=threshold)
        print(f"Threshold: {threshold} -> Cross-validation accuracy: {round(accuracy, config.ROUND_DIGITS)}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    if best_threshold is None:
        print("There is no best threshold")
        return None, None

    tree = treepredict.buildtree(train, scoref=scoref, beta=best_threshold)
    treepredict.prune(tree, best_threshold)
    test_accuracy = get_accuracy(tree, test)
    print(f"Test method gave an accuracy: {round(test_accuracy, config.ROUND_DIGITS)}")

    return best_threshold, best_accuracy


def test_find_best_threshold(filename) -> None:
    config.print_line(f"Finding best threshold for {filename} file")
    headers, data = treepredict.read(filename)
    best_threshold, best_accuracy = find_best_threshold(data)
    print(f"RESULT -> Best threshold: {best_threshold} with a CV accuracy: {round(best_accuracy, config.ROUND_DIGITS)}")


if __name__ == "__main__":
    if config.EXECUTE_FILE1:
        test_find_best_threshold(config.FILE1)  # decision_tree_example.txt

    if config.EXECUTE_FILE2:
        test_find_best_threshold(config.FILE2)  # iris.csv
