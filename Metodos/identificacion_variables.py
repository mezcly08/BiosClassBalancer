from imports import Imports

def identificar_variables(df, umbral_unicos=4):
    """
    Identifica variables categóricas y continuas en un DataFrame.
    Variables categóricas son aquellas que tienen menos de 'umbral_unicos' valores únicos
    o están específicamente identificadas como categóricas.
    """
    variables_categoricas = []
    variables_continuas = []

    for columna in df.columns:
        valores_unicos = df[columna].dropna().unique()
        n_valores_unicos = len(valores_unicos)

        # Identificar variables categóricas
        if n_valores_unicos <= umbral_unicos or df[columna].dtype == 'object':
            variables_categoricas.append(columna)
        else:
            variables_continuas.append(columna)

    return variables_categoricas, variables_continuas


#CODIFICACION DE CATEGORIAS
def revisar_codificacion_categoricas(df, variables_categoricas):
    for var in variables_categoricas:
        if Imports.pd.api.types.is_numeric_dtype(df[var]):
            continue  # Si ya está codificada, no se debe hacer nada
        else:
            encoder = Imports.OrdinalEncoder()
            df[var] = encoder.fit_transform(df[[var]])
    return df