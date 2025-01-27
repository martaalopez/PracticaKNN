
class KNNRegressor:
    def __init__(self, n_splits=5, max_neighbors=31):
        self.n_splits = n_splits
        self.max_neighbors = max_neighbors
        self.best_model = None
        self.best_params = None

    def train_and_validate(self, X, y):
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        best_score = float('inf')
        best_params = {'n_neighbors': None, 'weights': None}

        for weights in ['uniform', 'distance']:
            for n_neighbors in range(1, self.max_neighbors):
                fold_mse = []
                knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

                for train_idx, test_idx in cv.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    knn.fit(X_train, y_train)
                    predictions = knn.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    fold_mse.append(mse)

                mean_mse = np.mean(fold_mse)
                if mean_mse < best_score:
                    best_score = mean_mse
                    best_params['n_neighbors'] = n_neighbors
                    best_params['weights'] = weights

        self.best_params = best_params
        self.best_model = KNeighborsRegressor(**best_params)
        self.best_model.fit(X, y)
        print(f"Best Parameters: {self.best_params}, Best MSE: {best_score:.4f}")

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.best_model, f)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def predict(self, X):
        return self.best_model.predict(X)


if __name__ == "__main__":
    # Ejemplo de uso
    file_path = 'winequality-red.csv'
    df = pd.read_csv(file_path)
    X = df.drop('quality', axis=1).values
    y = df['quality'].values

    # Escalar características
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clasificador KNN
    classifier = KNNClassifier()
    classifier.train_and_validate(X_scaled, y)
    classifier.save_model("knn_classifier.pkl")

    # Regresor KNN
    regressor = KNNRegressor()
    regressor.train_and_validate(X_scaled, y)
    regressor.save_model("knn_regressor.pkl")