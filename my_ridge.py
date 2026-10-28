import numpy as np

class MyRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.X_mean_ = None
        self.y_mean_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # 1. Calcular las medias usando solamente los datos de entrenamiento
        self.X_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y)

        # 2. Centrar las features y la etiqueta (la matriz de diseño no lleva columna de unos)
        X_centered = X - self.X_mean_
        y_centered = y - self.y_mean_

        m, n = X_centered.shape

        # 3. Construir la matriz identidad (excluyendo el intercepto)
        I = np.eye(n)

        # 4. Resolver la ecuación cerrada con Ridge
        # Ecuación: (X^T X + alpha * I) * theta = X^T y
        A = np.dot(X_centered.T, X_centered) + self.alpha * I
        b = np.dot(X_centered.T, y_centered)
        
        self.coef_ = np.linalg.solve(A, b)

        # 5. Recuperar el intercepto (theta_0)
        self.intercept_ = self.y_mean_ - np.dot(self.X_mean_, self.coef_)
        
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.dot(X - self.X_mean_, self.coef_) + self.intercept_

    def score(self, X, y):
        # Coeficiente de determinación R^2 idéntico a sklearn
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def rmse(self, X, y):
        # Raíz del error cuadrático medio (RMSE)
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))