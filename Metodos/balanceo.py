from imports import Imports
import pandas as pd

def apply_smote(X, y):
    X_train, X_test, y_train, y_test = Imports.train_test_split(X, y, test_size=0.3, random_state=42)
    smote = Imports.SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test


def apply_smote_borderline(X, y):
    X_train, X_test, y_train, y_test = Imports.train_test_split(X, y, test_size=0.3, random_state=42)
    smote_borderline = Imports.BorderlineSMOTE(random_state=42, kind='borderline-1')
    X_train_resampled, y_train_resampled = smote_borderline.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test

def apply_smote_nc(X, y, categorical_features):
    X_train, X_test, y_train, y_test = Imports.train_test_split(X, y, test_size=0.3, random_state=42)

    # Convertimos categorical_features en lista de booleanos si aún no lo es
    if isinstance(categorical_features[0], bool):
        smote_nc = Imports.SMOTENC(categorical_features=categorical_features, random_state=42)
    else:
        categorical_bool = [col in categorical_features for col in X_train.columns]
        smote_nc = Imports.SMOTENC(categorical_features=categorical_bool, random_state=42)

    X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)
    return X_train_resampled, X_test, y_train_resampled, y_test

def apply_random_under_sampling(X, y):
    X_train, X_test, y_train, y_test = Imports.train_test_split(X, y, test_size=0.3, random_state=42)
    under_sampler = Imports.RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)
    return X_train_resampled, X_test, y_train_resampled, y_test


#Evaluar el desbalance de datos
def calculate_mir(n_i):
    """
    Calcula el índice de desequilibrio general MIR.
    """
    return max(n_i) / min(n_i) if min(n_i) > 0 else Imports.np.inf

#Multiclase
def calculate_lrid(n_i, p_i, N):
    """
    Calcula el Likelihood Ratio Imbalance Degree (LRID).
    """
    lrid = -2 * sum([n * Imports.np.log(n / (N * p)) for n, p in zip(n_i, p_i)])
    return 0 if lrid == 0 else abs(lrid)

def medir_desbalance(df, target_column):
    """
    Evalúa el nivel de desbalance de clases en un dataset dado una columna objetivo.
    """
    nivel = ""
    class_counts = df[target_column].value_counts().to_dict()
    n_i = list(class_counts.values())
    N = sum(n_i)
    C = len(class_counts)
    p_i = [1/C] * C  # Distribución uniforme esperada

    mir = calculate_mir(n_i)
    lrid = calculate_lrid(n_i, p_i, N)

    if mir <= 1.5:
        nivel = "Bajo"
    elif mir <= 3:
        nivel = "Medio"
    else:
        nivel = "Alto"

    return mir, lrid, nivel

def balancear_datasets(dataset, target, variables_categoricas, variables_continuas, name,app):
    """
    Aplica técnicas de balanceo a los datasets y guarda los resultados.
    """
    # Convertimos categorical_features en una lista booleana para `SMOTENC`
    categorical_features_bool = [col in variables_categoricas for col in dataset.columns if col != target]

    balance_methods = {
        "smote": lambda X, y: apply_smote(X, y),
        "smote_borderline": lambda X, y: apply_smote_borderline(X, y),
        "smote_nc": lambda X, y: apply_smote_nc(X, y, categorical_features_bool),
        "random_under": lambda X, y: apply_random_under_sampling(X, y)
    }

    results = []

    X = dataset.drop(columns=[target])
    y = dataset[target]
    graficar_proporcion_clases(y,'Datos_crudos')
    for method_name, balance_func in balance_methods.items():

        X_resampled, X_test, y_resampled, y_test = balance_func(X, y)

        df_balanced = Imports.pd.DataFrame(X_resampled, columns=X.columns)
        df_balanced[target] = y_resampled

        # Reconstruimos el dataset de prueba (sin balancear)
        df_test = Imports.pd.DataFrame(X_test, columns=X.columns)
        df_test[target] = y_test
    
        train_path = Imports.os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_{method_name}.csv")
        test_path = Imports.os.path.join(app.config['UPLOAD_FOLDER'], f"base_test.csv")

        df_balanced.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        mir, lrid, nivel = medir_desbalance(df_balanced, target)
        graficar_proporcion_clases(df_balanced[target], method_name)
        # Almacenar resultado en la lista
        results.append([method_name, mir, lrid, nivel])

    return results

#Modelo
def entrenar_y_evaluar_modelos(df, target, nombreMetodo,app):
    """
    Entrena modelos de XGBoost y KNN en el dataset dado y calcula métricas de evaluación.

    Args:
        df (DataFrame): Dataset balanceado.
        target (str): Nombre de la variable objetivo.
        metodo (str): Nombre del método de balanceo aplicado.

    Returns:
        dict: Métricas de evaluación para cada modelo.
    """
    # Separar variables predictoras (X) y la variable objetivo (y)
    X = df.drop(columns=[target])
    y = df[target]

    # Dividir en conjuntos de entrenamiento (60%) y prueba (40%)
    X_train, X_test, y_train, y_test = Imports.train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    # Definir los modelos
    modelos = {
        "XGBoost": Imports.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42),
        "KNN": Imports.KNeighborsClassifier(n_neighbors=5)
    }
    resultados = {}

    for nombre_modelo, modelo in modelos.items():
        # Entrenar el modelo
        modelo.fit(X_train, y_train)

        # Predecir en el conjunto de prueba
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

        # Calcular métricas
        accuracy = Imports.accuracy_score(y_test, y_pred)
        f1 = Imports.f1_score(y_test, y_pred, average='weighted')
        sensibilidad = Imports.recall_score(y_test, y_pred, average='weighted')
        matriz_confusion = Imports.confusion_matrix(y_test, y_pred)
        mcc = Imports.matthews_corrcoef(y_test, y_pred)
        balanced_accuracy_score = Imports.balanced_accuracy_score(y_test, y_pred)
        gmean = Imports.np.sqrt(Imports.recall_score(y_test, y_pred, pos_label=1) * Imports.recall_score(y_test, y_pred, pos_label=0))

        # Curva ROC y AUC
        if y_proba is not None:
            fpr, tpr, _ = Imports.roc_curve(y_test, y_proba)
            roc_auc = Imports.auc(fpr, tpr)
        else:
            roc_auc = None
            fpr = None
            tpr = None

        resultados[nombre_modelo] = {
                    "Accuracy": accuracy,
                    "F1-score": f1,
                    "Recall": sensibilidad,
                    "Matriz de Confusión": matriz_confusion.tolist(),
                    "MCC": mcc,
                    "AUC": roc_auc,
                    "Balanced_accuracy_score": balanced_accuracy_score,
                    "G-mean": gmean
                }
        print(resultados)
        df_predicciones = X_test.copy()
        df_predicciones['True_Label'] = y_test
        df_predicciones['Predicted_Label'] = y_pred

        # Guardar el dataset en un archivo CSV
        nombre = Imports.os.path.join(app.config['UPLOAD_FOLDER'], nombreMetodo+'_'+ nombre_modelo +'.csv')
        df_predicciones.to_csv(nombre, index=False)
            
    return resultados
    
def imagenesModelo(df, target, metodo, nombre):
    """
    Entrena modelos de XGBoost y KNN en el dataset dado y calcula métricas de evaluación.

    Args:
        df (DataFrame): Dataset balanceado.
        target (str): Nombre de la variable objetivo.
        metodo (str): Nombre del método de balanceo aplicado.

    Returns:
        dict: Métricas de evaluación para cada modelo.
    """
    
    # Separar variables predictoras (X) y la variable objetivo (y)
    X = df.drop(columns=[target])
    y = df[target]

    # Dividir en conjuntos de entrenamiento (70%) y prueba (30%)
    X_train, X_test, y_train, y_test = Imports.train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    modelo = None
    imagenes = []
    # Definir los modelos
    """modelos = {
        "XGBoost": Imports.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42),
        "KNN": Imports.KNeighborsClassifier(n_neighbors=5)
    }"""
    if(metodo == "XGBoost"):
        modelo = Imports.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
    else:
        modelo = Imports.KNeighborsClassifier(n_neighbors=5)

    # Entrenar el modelo
    modelo.fit(X_train, y_train)
    # Predecir en el conjunto de prueba
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

    # Calcular métricas
    accuracy = Imports.accuracy_score(y_test, y_pred)
    f1 = Imports.f1_score(y_test, y_pred, average='weighted')
    sensibilidad = Imports.recall_score(y_test, y_pred, average='weighted')
    matriz_confusion = Imports.confusion_matrix(y_test, y_pred)
    mcc = Imports.matthews_corrcoef(y_test, y_pred)
    gmean = Imports.np.sqrt(Imports.recall_score(y_test, y_pred, pos_label=1) * Imports.recall_score(y_test, y_pred, pos_label=0))
    # Curva ROC y AUC
    if y_proba is not None:
        fpr, tpr, _ = Imports.roc_curve(y_test, y_proba)
        roc_auc = Imports.auc(fpr, tpr)
    else:
        roc_auc = None
        fpr = None
        tpr = None

    if roc_auc is not None:
        Imports.plt.figure(figsize=(8, 6))
        Imports.plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.6f})')
        Imports.plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        Imports.plt.xlim([0.0, 1.0])
        Imports.plt.ylim([0.0, 1.05])
        Imports.plt.xlabel('False Positive Rate')
        Imports.plt.ylabel('True Positive Rate')
        Imports.plt.title(f'{nombre} - Curva ROC - {metodo}')
        Imports.plt.legend(loc='lower right')
        # Guardar la gráfica en memoria y convertirla a base64
        img = Imports.io.BytesIO()
        Imports.plt.savefig(img, format='png', bbox_inches="tight")
        img.seek(0)
        img_base64 = Imports.base64.b64encode(img.getvalue()).decode('utf-8')
        Imports.plt.close()
        imagenes.append(img_base64)
           
                  # Cerrar la figura para liberar memoria   
    return imagenes    

def nombreGanador(name,reemplazo):
    name = name.replace(reemplazo+"_", "")
    nombre = None
    if(name == 'SMOTE'):   
        nombre = 'Smote'
    elif(name == 'Borderline'):  
        nombre = 'Smote Borderline'
    elif(name == 'NC'):   
        nombre = 'SmoteNC'
    elif(name == 'RandomUnder'):
        nombre = 'Random Under-Sampling'
    return nombre

def seleccionar_mejor_modelo(resultados, nombresplit):
    """
    Selecciona los mejores modelos basándose en las métricas, devolviendo una lista si hay empate.

    Args:
        resultados (dict): Diccionario con los resultados de las métricas de cada modelo.

    Returns:
        tuple: El nombre del mejor método, la lista de los mejores modelos con su método, y el mejor puntaje según las métricas.
    """
    mejor_modelo = None
    mejor_score = -1
    mejor_metodo = None
    modelos_empate = []
    mensajes = []
    for metodo, modelos in resultados.items():
        for nombre_modelo, metrics in modelos.items():
            # Paso 1: Compara Balanced Accuracy Score
            score = metrics['Balanced_accuracy_score']
            
            if score > mejor_score:
                mejor_score = score
                mejor_modelo = nombre_modelo
                mejor_metodo = metodo
                modelos_empate = [{'metodo': metodo, 'modelo': nombre_modelo}]
            
            elif score == mejor_score:
                # Paso 2: Desempate con F1-Score
                f1_score = metrics['F1-score']
                mejor_f1 = resultados[mejor_metodo][mejor_modelo]['F1-score']
                
                if f1_score > mejor_f1:
                    mejor_modelo = nombre_modelo
                    mejor_metodo = metodo
                    modelos_empate = [{'metodo': metodo, 'modelo': nombre_modelo}]
                
                elif f1_score == mejor_f1:
                    # Paso 3: Desempate con G-Mean
                    gmean = metrics['G-mean']
                    mejor_gmean = resultados[mejor_metodo][mejor_modelo]['G-mean']
                    
                    if gmean > mejor_gmean:
                        mejor_modelo = nombre_modelo
                        mejor_metodo = metodo
                        modelos_empate = [{'metodo': metodo, 'modelo': nombre_modelo}]
                    
                    elif gmean == mejor_gmean:
                        # Paso 4: Desempate con MCC
                        mcc = metrics['MCC']
                        mejor_mcc = resultados[mejor_metodo][mejor_modelo]['MCC']
                        
                        if mcc > mejor_mcc:
                            mejor_modelo = nombre_modelo
                            mejor_metodo = metodo
                            modelos_empate = [{'metodo': metodo, 'modelo': nombre_modelo}]
                        
                        elif mcc == mejor_mcc:
                            # Paso 5: Desempate con AUC
                            auc = metrics['AUC']
                            mejor_auc = resultados[mejor_metodo][mejor_modelo]['AUC']
                            
                            if auc > mejor_auc:
                                mejor_modelo = nombre_modelo
                                mejor_metodo = metodo
                                modelos_empate = [{'metodo': metodo, 'modelo': nombre_modelo}]
                            
                            elif auc == mejor_auc:
                                # Paso 6: Desempate con Sensibilidad (Recall)
                                sensibilidad = metrics['Recall']
                                mejor_sensibilidad = resultados[mejor_metodo][mejor_modelo]['Recall']
                                
                                if sensibilidad > mejor_sensibilidad:
                                    mejor_modelo = nombre_modelo
                                    mejor_metodo = metodo
                                    modelos_empate = [{'metodo': metodo, 'modelo': nombre_modelo}]
                                
                                elif sensibilidad == mejor_sensibilidad:
                                    # Paso 7: Empate total
                                    modelos_empate.append({'metodo': metodo, 'modelo': nombre_modelo})

    for item in modelos_empate:
        nombremetodoganador = nombreGanador(item['metodo'], nombresplit)
        mensajes.append(item['modelo'] + "-" + nombremetodoganador)
    return mejor_metodo, modelos_empate, mejor_score,mensajes


def distribucion_por_columna(df, nombre_dataset):
    distribuciones = {}

    for col in df.columns:
        # Contar las instancias por clase (valor) en la columna
        class_distribution = df[col].value_counts()

        # Graficar la distribución en horizontal
        class_distribution.plot(kind='barh', color='skyblue', edgecolor='black')

        Imports.plt.title(f'Distribución de la columna "{col}" - {nombre_dataset}')
        Imports.plt.ylabel('Clase')  # ← Las clases ahora están en el eje Y
        Imports.plt.xlabel('Número de Instancias')  # ← Las cantidades están en el eje X

        # Guardar imagen en buffer
        img = Imports.io.BytesIO()
        Imports.plt.savefig(img, format='png', bbox_inches="tight")
        img.seek(0)

        # Codificar en base64
        img_base64 = Imports.base64.b64encode(img.getvalue()).decode('utf-8')
        Imports.plt.close()

        # Guardar en diccionario con el nombre de la columna
        distribuciones[col] = img_base64

    return distribuciones

def graficar_proporcion_clases(y, metodo):
    total = len(y)
    dist = Imports.Counter(y)
    clases = list(map(str, dist.keys()))
    proporciones = [v / total * 100 for v in dist.values()]
    valores = list(dist.values())

    # Crear gráfica
    Imports.plt.figure(figsize=(7, 5))
    barras = Imports.plt.bar(clases, proporciones, color='skyblue', edgecolor='black')

    Imports.plt.xlabel('Clase', fontsize=12)
    Imports.plt.ylabel('Proporción (%)', fontsize=12)
    Imports.plt.title(f'Proporción de Clases - {metodo}', fontsize=14, fontweight='bold')

    # Ajuste de límites para evitar que el texto se corte
    max_y = max(proporciones)
    Imports.plt.ylim(0, max_y + 10)

    # Anotaciones: valor absoluto dentro de la barra, porcentaje encima
    for i, (bar, prop, val) in enumerate(zip(barras, proporciones, valores)):
        altura = bar.get_height()

        # Valor absoluto centrado dentro de la barra
        Imports.plt.text(bar.get_x() + bar.get_width() / 2, altura / 2, f"{val}", 
                         ha='center', va='center', fontsize=11, fontweight='bold', color='black')

        # Porcentaje encima
        Imports.plt.text(bar.get_x() + bar.get_width() / 2, altura + 1.5, f"{prop:.2f}%", 
                         ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    Imports.plt.tight_layout()

    # Guardar en carpeta local como PNG
    nombre_archivo = f"{metodo}_proporcion.png"
    ruta_completa = Imports.os.path.join('static/uploads/', nombre_archivo)
    Imports.plt.savefig(ruta_completa, format='png', bbox_inches="tight")

    Imports.plt.close()



