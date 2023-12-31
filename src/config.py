##### treepredict #####
FILE1 = "decision_tree_example.txt"
FILE2 = "iris.csv"

##### evaluation #####
ROUND_DIGITS = 3
evaluation_thresholds = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]

##### clusters #####
FILE3 = "blogdata.txt"
FILE4 = "blogdata_full.txt"


def print_line(header="", length=60):
    print("-" * (length // 2), end="")
    print(f" {header} ", end="")
    print("-" * (length // 2))
