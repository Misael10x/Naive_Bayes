from flask import Flask, render_template, request
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import io
import base64


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calcular media, varianza y prior para cada clase
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # Calcular probabilidad posterior para cada clase
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # Retornar la clase con la mayor probabilidad posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = None
    accuracy = None

    if request.method == "POST":
        # Crear datos simulados
        X, y = datasets.make_classification(
            n_samples=1000,
            n_features=2,  # Total features
            n_informative=2,  # Número de características informativas
            n_redundant=0,  # Número de características redundantes
            n_clusters_per_class=1,
            random_state=123
        )

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        # Entrenar modelo Naive Bayes
        nb = NaiveBayes()
        nb.fit(X_train, y_train)
        predictions = nb.predict(X_test)

        # Calcular precisión
        accuracy = np.sum(y_test == predictions) / len(y_test)

        # Gráfica
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap="viridis", alpha=0.7, edgecolor="k")
        plt.title("Clasificación con Naive Bayes")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="Clase")

        # Convertir gráfica a imagen
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

    return render_template("index.html", plot_url=plot_url, accuracy=accuracy)


if __name__ == "__main__":
    app.run(debug=True)
