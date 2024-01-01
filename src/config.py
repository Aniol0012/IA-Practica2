############# treepredict #############
EXECUTE_FILE1 = False
FILE1 = "decision_tree_example.txt"

EXECUTE_FILE2 = True
FILE2 = "iris.csv"

# Beta value for pruning, if the increase of entropy is less than beta, the node will be pruned
beta = 0.1

############# evaluation #############
ROUND_DIGITS = 3
evaluation_thresholds = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]

# Percentage of the dataset that will be used for testing
test_size = 0.2  # 20%

# Number of partitions (folds) that will be generated
k = 5

# The seed used to separate the dataset
seed = 50

############# clusters #############
# Number of digits to round the distances output
ROUND_DIGITS_DISTANCES_CLUSTERS = 8

# Number of iterations for the k-means
iterations = 10

# Number of clusters to find
k_for_clusters = 4

# Range of values for each centroid, remember that range is [start, end)
k_range = range(2, 10)

EXECUTE_FILE3 = False
FILE3 = "blogdata.txt"

EXECUTE_FILE4 = True
FILE4 = "blogdata_full.txt"


############# general config #############
def print_line(header="", length=60):
    print("-" * (length // 2), end="")
    print(f" {header} ", end="")
    print("-" * (length // 2))
