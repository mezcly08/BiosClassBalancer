from imports import Imports
import matplotlib
matplotlib.use("Agg") 

#METODOS DE IMPUTACION

#GRADIENTE BOOSTING
def imputar_xgboost(df, variables_continuas, variables_categoricas):
    for variable in variables_continuas + variables_categoricas:
        if df[variable].isnull().sum() > 0:
            X = df.drop(columns=[variable]).select_dtypes(include=[float, int]).fillna(-1)
            y = df[variable]

            X_train, X_test, y_train, y_test = Imports.train_test_split(
                X[y.notnull()], y.dropna(), test_size=0.2, random_state=42
            )
            X_missing = X.loc[y.isnull()]

            if variable in variables_continuas:
                model = Imports.XGBRegressor(n_estimators=100, random_state=42)
            else:
                model = Imports.XGBClassifier(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_missing)
            df.loc[df[variable].isnull(), variable] = y_pred
    return df

#RANDOM FOREST
def imputar_random_forest(df, variables_continuas, variables_categoricas):
    for variable in variables_continuas + variables_categoricas:
        if df[variable].isnull().sum() > 0:
            X = df.drop(columns=[variable]).select_dtypes(include=[float, int]).fillna(-1)  # Llenar NaN temporales
            y = df[variable]
            # Separar datos con y sin valores nulos
            X_train, y_train = X.loc[y.notnull()], y.dropna()
            X_missing = X.loc[y.isnull()]

            # Escoger modelo basado en el tipo de variable
            if variable in variables_continuas:
                model = Imports.RandomForestRegressor(n_estimators=100, random_state=0)
            else:
                model = Imports.RandomForestClassifier(n_estimators=100, random_state=0)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_missing)
            df.loc[df[variable].isnull(), variable] = y_pred
    return df

#CONTINUAS

#KNeighborsRegressor (supervisado)
def imputar_knn_regresion(df, variables_continuas, k=5):
    """
    Imputa valores faltantes en variables continuas usando KNN.

    Parámetros:
    - df: DataFrame con los datos.
    - variables_continuas: Lista de variables continuas a imputar.
    - k: Número de vecinos a considerar en KNN (default=5).

    Retorna:
    - df con valores imputados en variables continuas.
    """
    for target_variable in variables_continuas:
        if df[target_variable].isnull().sum() > 0:
            matrizX = df[variables_continuas].drop(columns=[target_variable])
            SerieY = df[target_variable].dropna()
            X_train = matrizX.loc[SerieY.index].fillna(0)

            # Entrenar modelo KNN
            model = Imports.KNeighborsRegressor(n_neighbors=k)
            model.fit(X_train, SerieY)

            # Predecir valores faltantes
            df_missing = df.loc[df[target_variable].isnull()]
            y_pred = model.predict(df_missing[variables_continuas].drop(columns=[target_variable]).fillna(0))

            # Calcular MSE para evaluar la predicción
            mse = Imports.mean_squared_error(SerieY, model.predict(X_train))

            # Imputar si el error es aceptable
            if mse < 0.1:
                df.loc[df[target_variable].isnull(), target_variable] = y_pred
            else:
                knn_imputer = Imports.KNNImputer(n_neighbors=k)
                df[variables_continuas] = knn_imputer.fit_transform(df[variables_continuas])

    return df

#KNN (no supervisado)
def imputar_knn(df, variables_continuas):
    pipeline_continuas = Imports.make_pipeline(Imports.KNNImputer(n_neighbors=5, weights='uniform'))
    df[variables_continuas] = pipeline_continuas.fit_transform(df[variables_continuas])
    return df

#REGRESION LINEAL
def imputar_regresion(df, variables_continuas):
    for target_variable in variables_continuas:
        if df[target_variable].isnull().sum() > 0:
            matrizX = df[variables_continuas].drop(columns=[target_variable]).fillna(0)
            SerieY = df[target_variable].dropna()
            X_train = matrizX.loc[SerieY.index]
            model = Imports.LinearRegression()
            model.fit(X_train, SerieY)
            df_missing = df.loc[df[target_variable].isnull()]
            y_pred = model.predict(df_missing[variables_continuas].drop(columns=[target_variable]).fillna(0))
            mse = Imports.mean_squared_error(SerieY, model.predict(X_train))
            if mse < 0.1:
                df.loc[df[target_variable].isnull(), target_variable] = y_pred
            else:
                df[target_variable] = imputar_knn(df, [target_variable])
    return df

#MEDIANA

def imputar_mediana(df, variables_continuas):
    for var in variables_continuas:
        mediana = df[var].median()
        df[var] = df[var].fillna(mediana)
    return df


#CATEGORICAS

def imputar_moda(df, variables_categoricas):
    for var in variables_categoricas:
        moda = df[var].mode()[0]
        df[var] = df[var].fillna(moda)
    return df


# Imputacion simple (Kernel)
def imputar_categoricas_simple(df, variables_categoricas):
    """
    Imputa valores categóricos faltantes usando la categoría más frecuente (moda).
    Es un método no supervisado y simple.

    Parámetros:
    - df: DataFrame con los datos
    - variables_categoricas: Lista de columnas categóricas a imputar

    Retorna:
    - DataFrame con imputación aplicada
    """
    imputer = Imports.SimpleImputer(strategy="most_frequent")

    df[variables_categoricas] = imputer.fit_transform(df[variables_categoricas])

    return df


#Random Forest (supervisado)

def imputar_categoricas_random_forest(df, variables_categoricas):
    """
    Imputa valores categóricos faltantes utilizando RandomForestClassifier de forma supervisada.

    Parámetros:
    - df: DataFrame con los datos.
    - variables_categoricas: Lista de variables categóricas a imputar.

    Retorna:
    - DataFrame con imputación aplicada.
    """
    for variable in variables_categoricas:
        if df[variable].isnull().sum() > 0:
            # Seleccionar solo variables numéricas para predecir las categorías
            X = df.drop(columns=[variable], errors='ignore').select_dtypes(include=[Imports.np.number])
            X = X.fillna(X.median(numeric_only=True))

            # Variable objetivo (quitar valores nulos)
            y = df[variable].dropna()

            if len(y) == 0:
                continue

            # Dividir en datos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = Imports.train_test_split(
                X.loc[y.index], y, test_size=0.2, random_state=42
            )
            X_missing = X.loc[df[variable].isnull()]

            if X_missing.empty:
                continue

            # Entrenar modelo RandomForestClassifier
            model = Imports.RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predecir valores faltantes
            y_pred = model.predict(X_missing)

            df.loc[df[variable].isnull(), variable] = y_pred

    return df    

#Grafica de distribuciones generales



def graficar_distribuciones(df_original, df_xgboost, df_random_forest, df_knn, df_moda_mediana, variables_continuas):
    """
    Grafica la distribución de las variables continuas en el dataset original y las imputaciones
    """
    imagenes_base64 = []
    for var in variables_continuas:
        Imports.plt.figure(figsize=(10, 5))

        # Gráfico de distribución (KDE) para los datos originales
        Imports.sns.kdeplot(df_original[var].dropna(), label='Original', color='blue', linewidth=2)

        # Gráficos de distribución para cada método de imputación
        Imports.sns.kdeplot(df_xgboost[var].dropna(), label='XGBoost', color='green', linestyle='--', linewidth=2)
        Imports.sns.kdeplot(df_random_forest[var].dropna(), label='Random Forest', color='red', linestyle='--', linewidth=2)
        Imports.sns.kdeplot(df_knn[var].dropna(), label='KNN + Regresión + categoricas simple', color='purple', linestyle='--', linewidth=2)
        Imports.sns.kdeplot(df_moda_mediana[var].dropna(), label='Moda + Mediana', color='orange', linestyle='--', linewidth=2)

        Imports.plt.title(f"Distribución de {var}: Original vs Imputaciones")
        Imports.plt.xlabel(var)
        Imports.plt.ylabel("Densidad")
        Imports.plt.legend()
        Imports.plt.grid(True, linestyle='--', alpha=0.6)

        # Guardar la imagen en memoria como base64
        buffer = Imports.io.BytesIO()
        Imports.plt.savefig(buffer, format='png', bbox_inches='tight')
        
        buffer.seek(0)
        imagen_base64 = Imports.base64.b64encode(buffer.getvalue()).decode('utf-8')
        imagenes_base64.append(imagen_base64)  
        Imports.plt.close()
    return imagenes_base64    


def evaluar_calidad_imputacion(df_original, imputaciones, variables_continuas, variable_objetivo):
    """
    Evalúa la calidad de la imputación midiendo la similitud de la distribución de los datos originales
    y los imputados, y el rendimiento de un modelo predictivo.

    Parámetros:
    - df_original: DataFrame con los datos originales antes de la imputación.
    - imputaciones: Diccionario con DataFrames imputados. Ejemplo: {"XGBoost": df_xgboost, "RandomForest": df_rf}
    - variables_continuas: Lista de variables continuas a evaluar.
    - variable_objetivo: Nombre de la variable objetivo (para evaluación del modelo predictivo).

    Retorna:
    - Diccionario con las métricas de calidad para cada método de imputación.
    - El mejor método basado en el rendimiento del modelo predictivo y la menor distancia de Wasserstein.
    """

    resultados = {}

    for metodo, df_imputado in imputaciones.items():
        wasserstein_scores = []
        ks_scores = []
        mean_diff = []
        std_diff = []

        for var in variables_continuas:
            original = df_original[var].dropna()
            imputado = df_imputado[var].dropna()

            if len(imputado) == 0 or len(original) == 0:
                continue  # Saltar si no hay datos

            # Distancia de Wasserstein
            wasserstein_scores.append(Imports.wasserstein_distance(original, imputado))

            # Prueba de Kolmogorov-Smirnov
            ks_stat, _ = Imports.ks_2samp(original, imputado)
            ks_scores.append(ks_stat)

            # Diferencia en la media y desviación estándar
            mean_diff.append(abs(original.mean() - imputado.mean()))
            std_diff.append(abs(original.std() - imputado.std()))

        # Evaluación del modelo predictivo con cada dataset imputado
        X = df_imputado.drop(columns=[variable_objetivo])
        y = df_imputado[variable_objetivo]

        # Filtrar datos sin valores nulos en la variable objetivo
        X, y = X[y.notnull()], y.dropna()

        # División en datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = Imports.train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar modelo RandomForestClassifier
        modelo = Imports.RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Métricas del modelo
        accuracy = Imports.accuracy_score(y_test, y_pred)
        f1 = Imports.f1_score(y_test, y_pred, average='weighted')

        # Guardar resultados
        resultados[metodo] = {
            "Wasserstein Mean": Imports.np.mean(wasserstein_scores),
            "KS Mean": Imports.np.mean(ks_scores),
            "Mean Diff": Imports.np.mean(mean_diff),
            "Std Diff": Imports.np.mean(std_diff),
            "Modelo Accuracy": accuracy,
            "Modelo F1-score": f1
        }

    # Determinar el mejor método basado en la menor distancia de Wasserstein y el mejor rendimiento del modelo
    mejor_metodo = min(resultados, key=lambda k: (resultados[k]["Wasserstein Mean"], -resultados[k]["Modelo Accuracy"]))

    return resultados, mejor_metodo