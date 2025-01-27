import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pickle
import os

class KNNClassifier:
    
    def __init__(self, n_splits=5, max_neighbors=31):
        self.n_splits = n_splits
        self.max_neighbors = max_neighbors
        self.best_model = None
        self.best_params = None

    def train_and_validate(self, X, y):
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        best_score = 0
        best_params = {'n_neighbors': None, 'weights': None}

        for weights in ['uniform', 'distance']:
            for n_neighbors in range(1, self.max_neighbors):
                fold_accuracy = []
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

                for train_idx, test_idx in cv.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    knn.fit(X_train, y_train)
                    predictions = knn.predict(X_test)
                    acc = accuracy_score(y_test, predictions)
                    fold_accuracy.append(acc)

                mean_accuracy = np.mean(fold_accuracy)
                if mean_accuracy > best_score:
                    best_score = mean_accuracy
                    best_params['n_neighbors'] = n_neighbors
                    best_params['weights'] = weights

        self.best_params = best_params
        self.best_model = KNeighborsClassifier(**best_params)
        self.best_model.fit(X, y)
        print(f"Best Parameters: {self.best_params}, Best Accuracy: {best_score:.4f}")

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.best_model, f)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def predict(self, X):
        return self.best_model.predict(X)