#!/usr/bin/env python3
import sys
import collections
from math import log2
from typing import List, Tuple
import config

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

    set1 = []
    set2 = []

    for row in part:
        if split_function(row, column, value):
            set1.append(row)
        else:
            set2.append(row)
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


def calculate_gain(part, column, value, current_score, score_function) -> Tuple[float, Tuple[Data, Data]]:
    set1, set2 = divideset(part, column, value)
    set1_len = len(set1)
    set2_len = len(set2)

    if set1_len == 0 or set2_len == 0:
        return 0, (None, None)

    part_set1 = set1_len / len(part)
    gain = current_score - part_set1 * score_function(set1) - (1 - part_set1) * score_function(set2)
    return gain, (set1, set2)


def get_best_split(part, score_function, current_score) -> Tuple[float, Tuple[int, str], Tuple[Data, Data]]:
    best_gain = 0
    best_criteria = None
    best_sets = None
    columns = len(part[0]) - 1

    for column in range(columns):
        column_values = set()
        for row in part:
            column_values.add(row[column])

        for value in column_values:
            gain, sets = calculate_gain(part, column, value, current_score, score_function)
            if gain > best_gain:
                best_gain = gain
                best_criteria = (column, value)
                best_sets = sets

    return best_gain, best_criteria, best_sets


def buildtree(part: Data, scoref=entropy, beta=0) -> DecisionNode:
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """
    if len(part) == 0:
        return DecisionNode()

    current_score = scoref(part)

    # Set up some variables to track the best criteria
    best_gain, best_criteria, best_sets = get_best_split(part, scoref, current_score)

    if best_gain > beta:
        tb = buildtree(best_sets[0], scoref, beta)
        fb = buildtree(best_sets[1], scoref, beta)
        return DecisionNode(col=best_criteria[0], value=best_criteria[1], tb=tb, fb=fb)
    else:
        return DecisionNode(results=unique_counts(part))


def iterative_buildtree(part: Data, scoref=entropy, beta=0) -> DecisionNode:
    """
    t10: Define the iterative version of the function buildtree
    """
    if len(part) == 0:
        return DecisionNode(results=unique_counts(part))

    # stack: (data, parent_node, is_tb)
    stack = [(part, None, 0)]
    root = None

    while stack:
        data, parent_node, is_tb = stack.pop()
        if not data:
            continue

        current_score = scoref(data)
        best_gain, best_criteria, best_sets = get_best_split(data, scoref, current_score)

        if best_gain > beta:
            node = DecisionNode(col=best_criteria[0], value=best_criteria[1])
            stack.append((best_sets[0], node, True))
            stack.append((best_sets[1], node, False))
        else:
            node = DecisionNode(results=unique_counts(data))

        if parent_node:
            if is_tb:
                parent_node.tb = node
            else:
                parent_node.fb = node
        else:
            root = node
    return root


def classify(tree, values):
    if tree.results is not None:
        return tree.results

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


def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = config.FILE1

    # header, data = read(filename)
    # print_data(header, data)

    # print(unique_counts(data))

    # print(gini_impurity(data))
    # print(gini_impurity([]))
    # print(gini_impurity([data[0]]))

    # print(entropy(data))
    # print(entropy([]))
    # print(entropy([data[0]]))

    # test_buildtree(config.FILE1)  # decision_tree_example.txt
    test_buildtree(config.FILE2)  # iris.csv


def test_buildtree(filename, recursive=True, iterative=True) -> None:
    config.print_line(filename, 80)
    headers, data = read(filename)

    if recursive:
        config.print_line("Recursive build tree")
        tree = buildtree(data)
        print_tree(tree, headers)

    if iterative:
        config.print_line("Iterative build tree")
        tree = iterative_buildtree(data)
        print_tree(tree)


if __name__ == "__main__":
    main()
