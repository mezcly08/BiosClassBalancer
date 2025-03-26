from imports import Imports

def graficar_valores_atipicos(df, variables_continuas):
    """
    Genera gráficos de dispersión para visualizar valores atípicos en variables continuas.
    
    Args:
        df: DataFrame con los datos originales.
        variables_continuas: Lista de nombres de variables continuas.
    
    Returns:
        Lista de imágenes en formato base64 para mostrar en un carrusel HTML.
    """
    Imports.sns.set_theme(style="whitegrid")
    imagenes_base64 = []
    resultados = []

    for var in variables_continuas:
        Imports.plt.figure(figsize=(10, 5))
        Imports.sns.stripplot(x=[var] * len(df), y=df[var], color="skyblue", alpha=0.9)
        Imports.plt.title(f"Distribución de {var}", fontsize=16)
        Imports.plt.xlabel(var, fontsize=12)
        Imports.plt.ylabel("Valores", fontsize=12)
        Imports.plt.xticks([])
        Imports.plt.ylim(df[var].min(), df[var].max())
        Imports.plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Guardar la gráfica en memoria y convertirla a base64
        img = Imports.io.BytesIO()
        Imports.plt.savefig(img, format='png', bbox_inches="tight")
        img.seek(0)
        img_base64 = Imports.base64.b64encode(img.getvalue()).decode('utf-8')
        imagenes_base64.append(img_base64)
        Imports.plt.close()  # Cerrar la figura para liberar memoria

        resultados.append({
            "variable": var,
            "max": df[var].max(),
            "min": df[var].min()
        })

    return imagenes_base64, resultados

def estandarizar_tipos(df):
    """
    Convierte todos los valores en las columnas del DataFrame en un solo tipo de dato.
    - Convierte todas las columnas numéricas a float.
    - Convierte todas las columnas categóricas a string.
    """
    for columna in df.columns:
        if df[columna].dtype == 'object':
            df[columna] = df[columna].astype(str).str.strip()
        else:
            df[columna] = Imports.pd.to_numeric(df[columna], errors='coerce')
    return df

def tratar_valores_atipicos(df, config_rangos):
    """
    Reemplaza valores fuera de los rangos especificados con NaN.

    Args:
        df (DataFrame): El DataFrame con los datos.
        config_rangos (dict): Diccionario con las columnas y sus rangos permitidos {columna: (min, max)}.

    Returns:
        DataFrame con los valores atípicos reemplazados con NaN.
    """
    for columna, (min_val, max_val) in config_rangos.items():
        if columna in df.columns:
            df[columna] = df[columna].apply(lambda x: Imports.np.nan if not (min_val <= x <= max_val) else x)
    return df


def evaluar_valores_vacios(df):
    """
    Evalúa el nivel de valores vacíos en el dataset y los clasifica en tres niveles.
    También calcula el nivel global de valores vacíos en todo el dataset.
    """
    # Calcular el porcentaje de valores nulos por columna
    porcentaje_nulos = (df.isnull().sum() / len(df)) * 100
    # Clasificación por columna
    niveles = porcentaje_nulos.apply(lambda x: 'Bajo (<5%)' if x < 5 else ('Medio (5%-40%)' if x <= 40 else 'Alto (>40%)'))

    # Crear DataFrame con los resultados por columna
    resultado = Imports.pd.DataFrame({'Columna': df.columns, 'Porcentaje_Nulos': porcentaje_nulos, 'Nivel': niveles})
    resultado = resultado.reset_index(drop=True)

    # Calcular el porcentaje total de valores nulos en el dataset
    total_nulos = df.isnull().sum().sum()
    total_celdas = df.shape[0] * df.shape[1]
    porcentaje_total_nulos = (total_nulos / total_celdas) * 100

    # Determinar el nivel global del dataset
    if porcentaje_total_nulos < 5:
        nivel_global = "Bajo"
    elif porcentaje_total_nulos <= 40:
        nivel_global = "Medio"
    else:
        nivel_global = "Alto"

    return resultado, porcentaje_total_nulos, nivel_global    