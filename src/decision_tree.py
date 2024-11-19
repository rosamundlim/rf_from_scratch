import numpy as np

class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        feature (int): Feature index for splitting.
        threshold (float): Threshold for the feature.
        value: Class label if it's a leaf node.
        left (Node): Left subtree.
        right (Node): Right subtree.
    """
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature # Feature index for splitting
        self.threshold = threshold # Threshold for the feature
        self.value = value # Class label if it's a leaf node
        self.left = left # Left subtree
        self.right = right # Right subtree


class DecisionTree:
    """
    Implements a decision tree classifier.

    Attributes:
        max_depth (int): Maximum depth of the tree.

    Methods:
        fit(X, y, depth=0)
            Fits the decision tree to the given training data.

        find_best_split(X, y)
            Finds the best split for a feature based on Gini impurity.

        calculate_gini(left_labels, right_labels)
            Calculates the Gini impurity for a split.

        majority_class(labels)
            Determines the majority class in a set of labels.

        predict(X, tree)
            Predicts the class for a given input using the trained decision tree.
    """
    def __init__(self, max_depth: int =None):
        """
        Initializes a decision tree.

        Parameters:
            max_depth (int): Maximum depth of the tree.
        """
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: np.array, y: np.array, depth: int =0) -> Node:
        """
        Fits the decision tree to the given training data.
        .fit() does the following: 
        i) Looks through all features and thresholds of the dataset (X), finds the 
        threshold that results in the lowest gini impurity in the corresponding Y labels. 
        ii) Splits data recursively until maximum depth is reached, finds pure nodes, or fails
        to improve gini impurity. 

        Parameters:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target labels.
            depth (int): Current depth of the tree.

        Returns:
            Node: The root node of the decision tree.
        """
        if len(np.unique(y)) == 1:
            return Node(value=y[0]) # reached leaf node
        
        if self.max_depth is not None and depth == self.max_depth:
            return Node(value=self.majority_class(y))

        num_features = X.shape[1]
        best_feature = None
        best_threshold = None
        best_gini = float('inf')

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(y[right_indices]) == 0:
                    pass
                else:
                    gini = self.gini(y[left_indices], y[right_indices])
                    if gini < best_gini:
                        best_gini = gini
                        best_feature = feature
                        best_threshold = threshold

        if best_gini == float('inf'):
            return Node(value=self.majority_class(y))

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        left_tree = self.fit(X[left_indices, :], y[left_indices], depth + 1)
        right_tree = self.fit(X[right_indices, :], y[right_indices], depth + 1)

        self.tree = Node(feature=best_feature, threshold=best_threshold, left=left_tree, right=right_tree)

        return self.tree

    def gini(self, left_labels: np.array, right_labels: np.array) -> float:
        """
        Calculates the Gini impurity for a split.

        Parameters:
            left_labels (numpy.ndarray): Labels for the left split.
            right_labels (numpy.ndarray): Labels for the right split.

        Returns:
            float: Gini impurity.
        """
        total_samples = len(left_labels) + len(right_labels)
        p_left = len(left_labels) / total_samples
        p_right = len(right_labels) / total_samples
        _unused, class_count_left = np.unique(left_labels, return_counts=True)
        _unused, class_count_right = np.unique(right_labels, return_counts=True)
        gini_left = 1 - sum(((count/len(left_labels))**2 for count in class_count_left))
        gini_right = 1 - sum(((count/len(right_labels))**2 for count in class_count_right))
        gini_impurity = p_left * gini_left + p_right * gini_right
        return gini_impurity

    def majority_class(self, labels: np.array) -> int:
        """
        Determines the majority class in a set of labels.

        Parameters:
            labels (numpy.ndarray): Input labels.

        Returns:
            int: Majority class.
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        return unique_labels[np.argmax(counts)]
    
    def predict_one_sample(self, X: np.array, node: Node) -> np.array:
        """
        Returns the prediction probabilities for a single-dimensional (a row in the data) input array
        by traversing the tree. 

        Parameters:
            X (numpy.ndarray): Input array representing a single data sample.

        Returns:
            numpy.ndarray: Prediction probabilities for each class.
        """
        current_node = node

        while current_node.value is None: # If node.value is None, we have not reached the leaf
            if X[current_node.feature] <= current_node.threshold: # If it is < threshold traverse to the left child
                current_node = current_node.left
            else:
                current_node = current_node.right

        # While loop ends when we reach a terminal leaf. Current node points to terminal leaf and 
        # we can extract the prediction using current_node.value
        
        return current_node.value


    def predict(self, X: np.array, tree: Node) -> int:
        """
        Predicts the class for a given input using the trained decision tree.

        Parameters:
            X (numpy.ndarray): Input features.
            tree (Node): The root node of the decision tree.

        Returns:
            numpy.ndarry: The predicted class.
        """
        
        y_pred = np.apply_along_axis(self.predict_one_sample, axis=1, arr=X, node=tree )

        return y_pred
