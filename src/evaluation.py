import random
from typing import Union, List


def train_test_split(dataset, test_size: Union[float, int], seed=None):
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


def get_accuracy(classifier, dataset):
    correct_parameter = 0
    for row in dataset:
        if classifier(row) == row[-1]:
            correct_parameter += 1
    return correct_parameter / len(dataset)


def mean(values: List[float]):
    return sum(values) / len(values)


def cross_validation(dataset, k, agg, seed, scoref, beta, threshold):
    if seed:
        random.seed(seed)

    # Divide the dataset into k folds
    folds = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / k)
    for _ in range(k):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        folds.append(fold)

    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        classifier = agg(train_set, scoref, beta, threshold)
        for row in test_set:
            row[-1] = classifier(row)
        actual = [row[-1] for row in fold]
        predicted = [row[-1] for row in test_set]
        accuracy = get_accuracy(actual, predicted)
        scores.append(accuracy)
    return mean(scores)
