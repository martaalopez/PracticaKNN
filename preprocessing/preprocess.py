import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

class Preprocesar:
    def __init__(self, data):
        self.data = data

    def detectar_outliers(self):
        isolation_forest = IsolationForest(random_state=42)
        num_data = self.data.select_dtypes(include=[np.number])
        outlier_pred = isolation_forest.fit_predict(num_data)
        self.data = self.data[outlier_pred == 1]

    def imputar_valores(self):
        imputer = SimpleImputer(strategy="median")
        num_data = self.data.select_dtypes(include=[np.number])
        imputed_data = imputer.fit_transform(num_data)
        self.data[num_data.columns] = imputed_data

    def escalar_datos(self):
        scaler = StandardScaler()
        num_data = self.data.select_dtypes(include=[np.number])
        scaled_data = scaler.fit_transform(num_data)
        self.data[num_data.columns] = scaled_data

    def normalizar_datos(self):
        normalizer = MinMaxScaler()
        num_data = self.data.select_dtypes(include=[np.number])
        normalized_data = normalizer.fit_transform(num_data)
        self.data[num_data.columns] = normalized_data

    def codificar_categorias(self):
        cat_data = self.data.select_dtypes(include=["object"])
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_data = encoder.fit_transform(cat_data)
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_data.columns))
        self.data = pd.concat([self.data.drop(cat_data.columns, axis=1), encoded_df], axis=1)

    def preprocesar_completo(self):
        self.detectar_outliers()
        self.insertar_columna()
        self.imputar_valores()
        self.escalar_datos()
        self.codificar_categorias()

if __name__ == "__main__":
    # Cargar los datos
    file_path = "winequality-red.csv"
    data = pd.read_csv(file_path)

    # Instanciar y utilizar la clase Preprocesar
    preprocesador = Preprocesar(data)
    preprocesador.normalizar_datos()  # Método específico para normalizar datos
    preprocesador.escalar_datos()     # Método específico para escalar datos
    preprocesador.preprocesar_completo()  # Método completo para preprocesar

    # Mostrar los primeros registros preprocesados
    print(preprocesador.data.head())
