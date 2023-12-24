#!/usr/bin/env python3
import sys
import collections
from math import log2
from typing import List, Tuple

import evaluation

# Used for typing
Data = List[List]


def read(file_name: str, separator: str = ",") -> Tuple[List[str], Data]:
    """
    t3: Load the data into a bidimensional list.
    Return the headers as a list, and the data
    """
    header = None
    data = []
    with open(file_name, "r") as fh:
        for line in fh:
            values = line.strip().split(separator)
            if header is None:
                header = values
                continue
            data.append([_parse_value(v) for v in values])
    return header, data


def _parse_value(v: str):
    try:
        if float(v) == int(v):
            return int(v)
        else:
            return float(v)
    except ValueError:
        return v
    # try:
    #     return float(v)
    # except ValueError:
    #     try:
    #         return int(v)
    #     except ValueError:
    #         return v


def unique_counts(part: Data):
    """
    t4: Create counts of possible results
    (the last column of each row is the
    result)
    """
    return dict(collections.Counter(row[-1] for row in part))

    # results = collections.Counter()
    # for row in part:
    #     c = row[-1]
    #     results[c] += 1
    # return dict(results)

    # results = {}
    # for row in part:
    #     c = row[-1]
    #     if c not in results:
    #         results[c] = 0
    #     results[c] += 1
    # return results


def gini_impurity(part: Data):
    """
    t5: Computes the Gini index of a node
    """
    total = len(part)
    if total == 0:
        return 0

    results = unique_counts(part)
    imp = 1
    for v in results.values():
        imp -= (v / total) ** 2
    return imp


def entropy(part: Data):
    """
    t6: Entropy is the sum of p(x)log(p(x))
    across all the different possible results
    """
    total = len(part)
    results = unique_counts(part)

    probs = (v / total for v in results.values())
    return -sum(p * log2(p) for p in probs)

    # imp = 0
    # for v in results.values():
    #     p = v / total
    #     imp -= p * log2(p)
    # return imp


def _split_numeric(prototype: List, column: int, value):
    return prototype[column] >= value


def _split_categorical(prototype: List, column: int, value: str):
    return prototype[column] == value


def divideset(part: Data, column: int, value) -> Tuple[Data, Data]:
    """
    t7: Divide a set on a specific column. Can handle
    numeric or categorical values
    """
    if isinstance(value, (int, float)):
        split_function = _split_numeric
    else:
        split_function = _split_categorical

    set1, set2 = [], []

    for row in part:  # For each row in the dataset
        if split_function(row, column, value):  # If it matches the criteria
            set1.append(row)  # Add it to the first set
        else:
            set2.append(row)  # Add it to the second set
    return set1, set2


class DecisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        """
        t8: We have 5 member variables:
        - col is the column index which represents the
          attribute we use to split the node
        - value corresponds to the answer that satisfies
          the question
        - tb and fb are internal nodes representing the
          positive and negative answers, respectively
        - results is a dictionary that stores the result
          for this branch. Is None except for the leaves
        """
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def buildtree(part: Data, scoref=entropy, beta=0):
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """
    if len(part) == 0:
        return DecisionNode()

    current_score = scoref(part)

    # Set up some variables to track the best criteria
    best_gain = 0
    best_criteria = None
    best_sets = None

    column_count = len(part[0]) - 1  # Attributes
    for column in range(column_count):
        column_values = set()
        for row in part:
            column_values.add(row[column])

        for value in column_values:
            set1, set2 = divideset(part, column, value)
            set1_len = len(set1)
            set2_len = len(set2)

            if set1_len == 0 or set2_len == 0:
                continue

            p = set1_len / len(part)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain:
                best_gain = gain
                best_criteria = (column, value)
                best_sets = (set1, set2)

    # Check if best gain is significant enough
    if best_gain > beta:
        tb = buildtree(best_sets[0], scoref, beta)
        fb = buildtree(best_sets[1], scoref, beta)
        return DecisionNode(col=best_criteria[0], value=best_criteria[1], tb=tb, fb=fb)
    else:
        return DecisionNode(results=unique_counts(part))


def iterative_buildtree(part: Data, scoref=entropy, beta=0):
    """
    t10: Define the iterative version of the function buildtree
    """
    if len(part) == 0:
        return DecisionNode(results=unique_counts(part))

    stack = []
    node_stack = []
    stack.append((0, part, None, 0))

    while stack:
        level, data, criteria, split_quality = stack.pop()
        if level == 0:
            current_score = scoref(data)
            if current_score == 0:
                node_stack.append(DecisionNode(results=unique_counts(data)))
            else:
                best_gain = 0.0
                best_criteria = None
                best_sets = None
                column_count = len(data[0]) - 1

                for column in range(column_count):
                    column_values = set()
                    for row in data:
                        column_values.add(row[column])
                    for value in column_values:
                        set1, set2 = divideset(data, column, value)
                        set1_len = len(set1)
                        set2_len = len(set2)
                        if set1_len == 0 or set2_len == 0:
                            continue
                        p = set1_len / len(data)
                        gain = get_gain(current_score, p, scoref, set1, set2)
                        if gain > best_gain:
                            best_gain = gain
                            best_criteria = (column, value)
                            best_sets = (set1, set2)
                if best_gain > beta:
                    stack.append((1, data, best_criteria, best_gain))
                    stack.append((0, best_sets[0], best_criteria, best_gain))
                    stack.append((0, best_sets[1], best_criteria, best_gain))
                else:
                    node_stack.append(DecisionNode(results=unique_counts(data)))
        elif level == 1:
            true_branch = node_stack.pop()
            false_branch = node_stack.pop()
            node_stack.append(DecisionNode(col=criteria[0], value=criteria[1], tb=true_branch, fb=false_branch))
            if len(data) == len(part):
                return node_stack.pop()
    return False


def get_gain(current_score, p, scoref, set1, set2):
    return current_score - p * scoref(set1) - (1 - p) * scoref(set2)


def classify(tree, values):
    if tree.results is not None:
        return tree.results
    else:
        v = values[tree.col]
        branch = None
        if isinstance(v, (int, float)):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(branch, values)


def print_tree(tree, headers=None, indent=""):
    """
    t11: Include the following function
    """
    # Is this a leaf node?
    if tree.results is not None:
        print(tree.results)
    else:
        # Print the criteria
        criteria = tree.col
        if headers:
            criteria = headers[criteria]
        print(f"{indent}{criteria}: {tree.value}?")

        # Print the branches
        print(f"{indent}T->")
        print_tree(tree.tb, headers, indent + "  ")
        print(f"{indent}F->")
        print_tree(tree.fb, headers, indent + "  ")


def print_data(headers, data):
    colsize = 15
    print('-' * ((colsize + 1) * len(headers) + 1))
    print("|", end="")
    for header in headers:
        print(header.center(colsize), end="|")
    print("")
    print('-' * ((colsize + 1) * len(headers) + 1))
    for row in data:
        print("|", end="")
        for value in row:
            if isinstance(value, (int, float)):
                print(str(value).rjust(colsize), end="|")
            else:
                print(value.ljust(colsize), end="|")
        print("")
    print('-' * ((colsize + 1) * len(headers) + 1))


""""
def test(data):
    print("Testing...")
    train = evaluation.train_test_split(data, 0.7)
    tree = buildtree(train)
    accuracy = evaluation.get_accuracy(lambda row: classify(tree, row), train)
    print("Accuracy on train dataset:", accuracy)
    accuracy = evaluation.get_accuracy(lambda row: classify(tree, row), data)
    print("Accuracy on test dataset:", accuracy)
"""


def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "decision_tree_example.txt"

    # header, data = read(filename)
    # print_data(header, data)

    # print(unique_counts(data))

    # print(gini_impurity(data))
    # print(gini_impurity([]))
    # print(gini_impurity([data[0]]))

    # print(entropy(data))
    # print(entropy([]))
    # print(entropy([data[0]]))

    headers, data = read(filename)
    tree = buildtree(data)
    # test(data)
    print_tree(tree, headers)


if __name__ == "__main__":
    main()
