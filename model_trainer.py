from imports import Imports

class ModelTrainer:
    def __init__(self, target_variable, feature_columns):
        """
        Inicializa el objeto ModelTrainer con la variable objetivo y las columnas de características.
        
        :param target_variable: Nombre de la columna objetivo (dependiente).
        :param feature_columns: Lista de nombres de las columnas de características (independientes).
        """
        self.target_variable = target_variable
        self.feature_columns = feature_columns
        self.model = Imports.RandomForestClassifier(random_state=42)

    def train_model(self, df):
        """
        Entrena un modelo de Random Forest usando el DataFrame proporcionado.
        
        :param df: DataFrame con los datos (debe contener la variable dependiente y las características).
        """
        # Separar la variable dependiente (y) y las características independientes (X)
        X = df[self.feature_columns]
        y = df[self.target_variable]
        
        # Entrenar el modelo
        self.model.fit(X, y)

    def get_feature_importances(self):
        """
        Obtiene la importancia de las características del modelo entrenado.
        
        :return: DataFrame con las importancias de las características ordenadas.
        """
        importances = self.model.feature_importances_
        feature_importances = Imports.pd.DataFrame(importances, index=self.feature_columns, columns=['importance']).sort_values('importance', ascending=False)
        return feature_importances

    def get_zero_importance_features(self):
        """
        Obtiene las variables cuya importancia es exactamente 0.0000000000.
        
        :return: Lista de nombres de las variables con importancia cero.
        """
        feature_importances = self.get_feature_importances()
        zero_importance_features = feature_importances[feature_importances['importance'] == 0.0].index.tolist()
        return zero_importance_features

    def plot_importances(self):
        """
        Crea una gráfica de barras de la importancia de las características con nombres legibles.
        
        :return: Imagen en base64 de la gráfica generada.
        """
        # Obtener la importancia de las características
        feature_importances = self.get_feature_importances()

        # Crear la figura con un tamaño mayor
        Imports.plt.figure(figsize=(12, 10))  # Aumentamos el tamaño vertical para más espacio
        Imports.sns.barplot(x=feature_importances.importance, y=feature_importances.index)
        
        Imports.plt.title("Importancia de las Características")
        Imports.plt.xlabel("Importancia")
        Imports.plt.ylabel("Características")

        # Asegurar que los nombres no se corten
        Imports.plt.yticks(rotation=0, ha="right")  # Rotación 0 y alineación a la derecha para legibilidad

        # Guardar la gráfica en un objeto BytesIO para convertirla a base64
        img = Imports.io.BytesIO()
        Imports.plt.savefig(img, format='png', bbox_inches="tight")  # bbox_inches evita el recorte de etiquetas
        img.seek(0)
        img_base64 = Imports.base64.b64encode(img.getvalue()).decode('utf-8')
        Imports.plt.close()  # Cerrar la figura para liberar memoria

        return img_base64
