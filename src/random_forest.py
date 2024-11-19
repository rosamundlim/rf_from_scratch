# Import libraries
import sys
import os
import numpy as np

# Add the project root folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.decision_tree import DecisionTree
except ImportError:
    print("Error importing DecisionTree from src.decision_tree")
    sys.exit(1)

# Main code

class RandomForest:
    """
    RandomForest is an ensemble learning model which builds multiple decision trees
    and aggregates their predictions to enhance performance, reducing overfitting. 

    This model is built using only Numpy. It uses bootstrapped samples for training
    individual trees and random feature subsets for splitting at each node. 

    Attributes:
        n_trees (int): No. of decision trees in the forest.
        max_depth (int): Max. depth of each tree in the forest. 
        subsample_size (float): Proportion of samples to use for training each tree.
        sample_with_replacement (bool): Whether to sample with replacement. Default is 1.
        feature_proportion (float): The proportion of features each tree is allowed to use; value between 0 and 1. 

    Methods:
        bootstrap(X, y): 
            Generates bootstrapped samples from the input data for training individual trees.
        
        subsampled_features(column_count):
            Randomly selects a subset of features for splitting in a decision tree. 
        
        fit(X, y):
            Trains the random forest by fitting decision trees to bootstrapped samples and subsampled features.
        
        predict(X, class_threshold=0.5):
            Predicts target labels for input features using majority voting across all decision trees. 
        
    """
    def __init__(self, n_trees: int = 5, max_depth: int = None, subsample_size = 1, 
                 sample_with_replacement = True, feature_proportion: float = 1.0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.subsample_size = subsample_size
        self.sample_with_replacement = sample_with_replacement
        self.feature_proportion = feature_proportion
        self.rforest = []
    
    def bootstrap(self, X: np.array, y: np.array) -> tuple: 
        """
        Generates bootstrapped samples for training a decision tree.

        Parameters:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target labels.

        Returns:
            tuple: Bootstrapped samples (X_bootstrap, y_bootstrap).
        """

        # Generate random indices
        indices = np.random.choice(len(X), size=int(len(X) * self.subsample_size), replace=self.sample_with_replacement)

        # Use random indices to get the bootstrapped samples
        X_bootstrap, y_bootstrap = X[indices], y[indices]   

        return X_bootstrap, y_bootstrap
    
    def subsampled_features(self, column_count: int) -> np.array:
        """Generate indices for feature subsampling

        Args:
            column_count (int): No. of features in the original dataset represented by the no. of columns.
        Returns:
            np.array: Indices of selected features
        """
        # Determine no. of features to be included in the subsample
        feature_count = int(column_count * self.feature_proportion)

        # Randomly sampled features without replacement
        sampled_features = np.random.choice(column_count, size=feature_count, replace=False)

        return sampled_features
    
    def fit(self, X: np.array, y: np.array):
        """Fits the random forest to given training data

        Args:
            X (np.array): input features.
            y (np.array): target labels.
        """
        for tree in range(self.n_trees):
            # Use bootstrapped samples to train each tree
            X_bootstrap, y_bootstrap = self.bootstrap(X,y)

            # Instantiate tree
            tree = DecisionTree(max_depth=self.max_depth)
            
            # Index X array based on sampled features
            selected_features = self.subsampled_features(X.shape[1])
            X_bs_fp = X_bootstrap[:, selected_features]

            # Fit tree
            tree.fit(X_bs_fp, y_bootstrap)

            # Append tree to forest 
            self.rforest.append((tree, selected_features))
        
    def predict(self, X: np.array, class_threshold: float = 0.5) -> np.array:
        """Predicts target label given input features

        Args:
            X (np.array): input features
            class_threshold (float, optional): Threshold to predict as positive class (target label =1). Defaults to 0.5.

        Returns:
            np.array: predicted labels that takes on either 1 or 0
        """
        # Make predictions with each tree
        predictions = np.array([dtree.predict(X[:,selected_features], dtree.tree).reshape(-1,1) for (dtree, selected_features) in self.rforest]) # dtree.predict is a method from DecisionTree class
        
        # Use simple averaging voting mechanism
        voting_rforest = np.mean(predictions, axis=0)

        # Convert averaged probabilites into class labels based on threshold that is set
        final_prediction = np.where(voting_rforest >= class_threshold, 1, 0)

        return final_prediction


        
        

    

