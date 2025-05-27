from imports import Imports
from flask import Flask, request, send_file, redirect, render_template, flash, jsonify, send_from_directory
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm

from model_trainer import ModelTrainer
from Metodos.identificacion_variables import identificar_variables, revisar_codificacion_categoricas
from Metodos.valores_atipicos import graficar_valores_atipicos, estandarizar_tipos, tratar_valores_atipicos, evaluar_valores_vacios
from Metodos.imputacion import imputar_xgboost,imputar_random_forest, imputar_knn_regresion, imputar_knn, imputar_regresion, imputar_mediana, imputar_moda, imputar_categoricas_simple, imputar_categoricas_random_forest, graficar_distribuciones, evaluar_calidad_imputacion
from Metodos.balanceo import medir_desbalance, balancear_datasets, entrenar_y_evaluar_modelos, imagenesModelo, seleccionar_mejor_modelo, distribucion_por_columna


app = Flask(__name__)
Bootstrap(app)  # Integra Bootstrap en Flask

# Configuración para la subida de archivos
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'tu_clave_secreta'  # Necesario para usar flash

df = None
dt = None
nombre_archivo = None
variables_categoricas = None
variables_continuas = None
dependiente = None
mejor_metodo = None

# Asegúrate de que la carpeta de subida exista

if not Imports.os.path.exists(UPLOAD_FOLDER):
    Imports.os.makedirs(UPLOAD_FOLDER)

def eliminar_archivo_si_existe(carpeta):
    if Imports.os.path.exists(carpeta):
        for archivo in Imports.os.listdir(carpeta):
                archivo_path = Imports.os.path.join(carpeta, archivo)
                
                # Eliminar archivo
                if Imports.os.path.isfile(archivo_path):
                    Imports.os.remove(archivo_path)
            
    
@app.route('/', methods=['GET', 'POST'])
def index():
    global df
    global nombre_archivo

    eliminar_archivo_si_existe(app.config['UPLOAD_FOLDER'])

    if request.method == 'POST':
        # Verifica si se ha subido un archivo
        if 'csv_file' not in request.files:
            flash('No se ha seleccionado ningún archivo.', 'error')
            return redirect(request.url)
        
        file = request.files['csv_file']
        
        # Si el usuario no selecciona un archivo, el navegador puede enviar un archivo vacío
        if file.filename == '':
            flash('No se ha seleccionado ningún archivo.', 'error')
            return redirect(request.url)
        
        if file:
            # Guarda el archivo en la carpeta de subida
            file_path = Imports.os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Lee el archivo CSV para obtener los nombres de las columnas
            try:
                # Cargar el archivo CSV
                df = Imports.pd.read_csv(file_path, na_values=['?', 'NULL', 'NaN','N/A','Unknown', Imports.np.nan])
                df.to_csv(file_path, index=False)
                # Obtener los nombres de las columnas como una lista
                columnas = df.columns.tolist()
                nombre_archivo = file.filename

                return render_template('index.html', columnas=columnas, archivo_subido=True, nombre_archivo=nombre_archivo)
            except Exception as e:
                flash(f'Error al leer el archivo CSV: {str(e)}', 'error')
                return redirect(request.url)

    return render_template('index.html', archivo_subido=False)

@app.route('/contacto')
def contacto():
    return render_template('contacto.html')

@app.route('/guardar_seleccion', methods=['POST'])
def guardar_seleccion():
    global dependiente
    global variables_continuas
    # Obtener la columna dependiente seleccionada
    dependiente = request.form['dependiente']

    variables_categoricas, variables_continuas = identificar_variables(df)
    revisar_codificacion_categoricas(df, variables_categoricas)

    # Separar las columnas
    X = df.drop([dependiente], axis=1, errors="ignore")
    y = df[dependiente]

    # Crear el objeto ModelTrainer
    model_trainer = ModelTrainer(dependiente, X.columns.tolist())

    # Entrenar el modelo y obtener la importancia de las características
    model_trainer.train_model(df)
    
    # Obtener las importancias de las características
    feature_importances = model_trainer.get_feature_importances()

    # Obtener las variables con importancia cero
    zero_importance_features = model_trainer.get_zero_importance_features()
    # Generar el gráfico de importancias
    plot_b64 = model_trainer.plot_importances()

    # Retornar las importancias y el gráfico
    return render_template('resultado.html', feature_importances=feature_importances.to_dict()['importance'], plot_b64=plot_b64, zero_importance_features=zero_importance_features)

@app.route('/guardar_variables', methods=['POST'])
def guardar_variables():
    global nombre_archivo
    global df
    global variables_categoricas
    global variables_continuas

    # Obtener las variables seleccionadas desde el formulario
    variables_seleccionadas = request.form.getlist('eliminar')
    df.drop(columns=variables_seleccionadas, inplace=True, errors='ignore')
    df = df.drop_duplicates()

    columnas = df.columns.tolist()

    file_path = Imports.os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo)
    Imports.os.remove(file_path)
    # Guardar el nuevo dataset
    df.to_csv(file_path, index=False)

    #IDENTIFICAR VARIABLES (CATEGORICAS Y CONTINUAS)
    variables_categoricas, variables_continuas = identificar_variables(df)
    
    # Redirigir a la página de resultados u otra vista
    return render_template('categorizacionVariables.html', columnas=columnas, variables_categoricas=variables_categoricas, variables_continuas=variables_continuas )

@app.route('/valores_atipicos', methods=['POST'])
def valores_atipicos():
    global df
    global variables_categoricas
    global variables_continuas

    variables_categoricas = request.form.getlist("variables_categoricas")
    variables_continuas = request.form.getlist("variables_continuas")
    
    df = revisar_codificacion_categoricas(df, variables_categoricas)
    

    imagenes, resultados = graficar_valores_atipicos(df, variables_continuas)

    return render_template('valoresAtipicos.html', imagenes=imagenes, resultados=resultados)

@app.route('/guardar_rangos', methods=['POST'])
def guardar_rangos():
    global df
    global nombre_archivo
    variables = request.form.getlist('variable[]')
    minimos = request.form.getlist('min[]')
    maximos = request.form.getlist('max[]')

    # Procesar los datos
    rangos_modificados = {
        variables[i]: (float(minimos[i]), float(maximos[i])) for i in range(len(variables))
    }
    df = estandarizar_tipos(df)
    df = tratar_valores_atipicos(df, rangos_modificados)
    

    resultado_valores_vacios, porcentaje_total_nulos, nivel_global = evaluar_valores_vacios(df)

    file_path = Imports.os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo)
    Imports.os.remove(file_path)

    file_path = Imports.os.path.join(app.config['UPLOAD_FOLDER'], "datasetBase.csv")
    # Guardar el nuevo dataset
    df.to_csv(file_path, index=False)

    resultado_valores_vacios = resultado_valores_vacios.to_dict(orient='records')
    # Redireccionar a una página de confirmación o recargar la página
    return render_template('panelImputacion.html', resultado_valores_vacios=resultado_valores_vacios, porcentaje_total_nulos = porcentaje_total_nulos, nivel_global = nivel_global)  

@app.route('/imputacion', methods=['POST'])
def imputacion():
    global df
    global variables_categoricas
    global variables_continuas
    global dependiente
    global mejor_metodo

    # Imputación con XGBoost
    df_xgboost = imputar_xgboost(df.copy(), variables_continuas, variables_categoricas)
    file_path_xgboost  = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_xgboost.csv')

    # Guardar dataset XGBoost
    df_xgboost.to_csv(file_path_xgboost, index=False)

    # Imputación con Random Forest
    df_random_forest = imputar_random_forest(df.copy(), variables_continuas, variables_categoricas)
    file_path_random_forest = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_random_forest.csv')

    # Guardar dataset Random Forest
    df_random_forest.to_csv(file_path_random_forest, index=False)

    # Imputación con KNNImputer y Regresión Lineal + categoricas simple
    df_knn = imputar_knn_regresion(df.copy(), variables_continuas)
    df_knn = imputar_regresion(df_knn, variables_continuas)
    df_knn = imputar_categoricas_random_forest(df_knn, variables_categoricas)
    file_path_knn = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_knn.csv')

    # Guardar dataset KNNImputer y Regresión Lineal + categoricas simple
    df_knn.to_csv(file_path_knn, index=False)

    # Imputación con Moda y Mediana
    df_moda_mediana = imputar_moda(df.copy(), variables_categoricas)
    df_moda_mediana = imputar_mediana(df_moda_mediana, variables_continuas)
    file_path_moda_mediana = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_moda_mediana.csv')
    # Guardar dataset Moda y Mediana
    df_moda_mediana.to_csv(file_path_moda_mediana, index=False)

    imputaciones = {
        "XGBoost": df_xgboost,
        "Random Forest": df_random_forest,
        "KNN + Regresión": df_knn,
        "Moda + Mediana": df_moda_mediana
    }
    resultados_imputacion, mejor_metodo = evaluar_calidad_imputacion(df, imputaciones, variables_continuas, dependiente)
    imagenes_base64 = graficar_distribuciones(df, df_xgboost, df_random_forest, df_knn, df_moda_mediana, variables_continuas) 
    return render_template('imputacion.html', imagenes_base64= imagenes_base64, resultados_imputacion = resultados_imputacion, mejor_metodo=mejor_metodo)

@app.route('/descargar/<filename>')
def descargarDataset(filename):

    if filename == 'XGBoost':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_xgboost.csv')
    elif filename == 'Random Forest':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_random_forest.csv')
    elif filename == 'KNNImputer':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_knn.csv')
    elif filename == 'Moda y Mediana':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_moda_mediana.csv')

    return send_from_directory(directory=Imports.os.path.dirname(directory),
                                   path=Imports.os.path.basename(directory),
                                   as_attachment=True)

@app.route('/visualizar/<filename>')
def visualizarDataset(filename):
    global df
    valores = df.columns.tolist()
    valorantiguo= df.isnull().sum().values.tolist()
    valornuevo = None

    if filename == 'XGBoost':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_xgboost.csv')
    elif filename == 'Random Forest':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_random_forest.csv')
    elif filename == 'KNNImputer':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_knn.csv')
    elif filename == 'Moda y Mediana':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_moda_mediana.csv')

    dr = Imports.pd.read_csv(directory)
    valornuevo = dr.isnull().sum().values.tolist()

    return [valores, valorantiguo, valornuevo]  

@app.route('/visualizarEstadistica/<filename>')
def visualizarEstadistica(filename):
    global df

    if filename == 'XGBoost':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_xgboost.csv')
    elif filename == 'Random Forest':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_random_forest.csv')
    elif filename == 'KNNImputer':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_knn.csv')
    elif filename == 'Moda y Mediana':
        directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_moda_mediana.csv')

    if(filename == 'base'):
        dr = df
    else:    
        dr = Imports.pd.read_csv(directory)

    descripcion = df.describe()
    estadisticas = []

    for i, columna in enumerate(descripcion.columns):
        estadisticas.append({
            "columna": columna,
            "count": descripcion.loc["count", columna],
            "mean": descripcion.loc["mean", columna],
            "min": descripcion.loc["min", columna],
            "25%": descripcion.loc["25%", columna],
            "50%": descripcion.loc["50%", columna],
            "75%": descripcion.loc["75%", columna],
            "max": descripcion.loc["max", columna]
        })
    return estadisticas


#-------------------------AQUI EMPIEZA EL BALANCEO DE DATOS-----------------
nombreDataset = None
modelo = None
imagenes = None
mir = None 
lrid = None 
nivel = None 
def datasetGanador():
    global mejor_metodo
    rutaGanador = None
    if mejor_metodo == 'XGBoost':
        rutaGanador = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_xgboost.csv')
    elif mejor_metodo == 'Random Forest':
        rutaGanador = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_random_forest.csv')
    elif mejor_metodo == 'KNN + Regresión':
        rutaGanador = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_knn.csv')
    elif mejor_metodo == 'Moda + Mediana':
        rutaGanador = Imports.os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_moda_mediana.csv')

    return rutaGanador

@app.route('/indexBalanceo')
def indexBalanceo():
    global dt
    global dependiente
    global variables_categoricas
    global variables_continuas
    global nombreDataset
    global mir
    global lrid
    global nivel

    directory = datasetGanador()
    dt = Imports.pd.read_csv(directory)
    nombreDataset = Imports.os.path.splitext(Imports.os.path.basename(directory))[0]
    variables_categoricas, variables_continuas = identificar_variables(dt)
    target = dependiente
    mir, lrid, nivel = medir_desbalance(dt, target)

    return render_template('balanceoDatosInicial.html', nombre=nombreDataset, mir=mir, lrid=lrid, nivel=nivel)

@app.route('/Balanceo')
def Balanceo():
    global dt
    global variables_categoricas
    global variables_continuas
    global nombreDataset
    global dependiente
    global mir
    global lrid
    global nivel
    
    directory = datasetGanador()
    dt = Imports.pd.read_csv(directory)
    target = dependiente
    resultados = balancear_datasets(dt, target, variables_categoricas, variables_continuas,nombreDataset,app)
    resultados.insert(0,['Datos crudos', mir, lrid, nivel])
    return render_template('resultadosBalanceo.html', resultados=resultados)


@app.route('/Modelo')
def Modelo():
    global nombreDataset
    global modelo
    global imagenes
    global dependiente
    global mejor_metodo

    imagenes=[]
    modelo = {}
    mensaje = []
    target = dependiente
    nombresplit = nombreDataset.replace("dataset_", "")


    datasets_cargados = {
        nombresplit + "_SMOTE": Imports.pd.read_csv(Imports.os.path.join(app.config['UPLOAD_FOLDER'], nombreDataset + '_smote.csv')),
        nombresplit + "_Borderline": Imports.pd.read_csv(Imports.os.path.join(app.config['UPLOAD_FOLDER'],nombreDataset + '_smote_borderline.csv')),
        nombresplit + "_NC": Imports.pd.read_csv(Imports.os.path.join(app.config['UPLOAD_FOLDER'],nombreDataset + '_smote_nc.csv')),
        nombresplit + "_RandomUnder": Imports.pd.read_csv(Imports.os.path.join(app.config['UPLOAD_FOLDER'],nombreDataset + '_random_under.csv')),
        "datos_crudos": Imports.pd.read_csv(Imports.os.path.join(app.config['UPLOAD_FOLDER'],'base_test.csv')),
    }
    
    for metodo, df in datasets_cargados.items():
        nombre = metodo.replace(nombresplit+"_", "")
        modelo[metodo] = entrenar_y_evaluar_modelos(df, target, nombre,app)
    
    modeloSinBase = modelo.copy()
    del modeloSinBase['datos_crudos']
    mejor_metodo1, mejor_modelo, mejor_score, mensajes = seleccionar_mejor_modelo(modeloSinBase, nombresplit)
    resultado = modelo.copy()
    return render_template('resultadosModelos.html', resultados = resultado, nombresplit=nombresplit, mensajes = mensajes, ganador= mensajes.copy())

def definirModelo(modelo,nombresplit):

    for clave in list(modelo.keys()):  
        name = clave.replace(nombresplit+"_", "")
        if name == 'Borderline':
            modelo['Smote Borderline'] = modelo.pop(clave)
        elif(name == 'SMOTE') :
            modelo['Smote'] = modelo.pop(clave)  
        elif(name == 'NC') :
            modelo['SmoteNC'] = modelo.pop(clave)  
        elif(name == 'RandomUnder') :
            modelo['Random Under-Sampling'] = modelo.pop(clave)   
    
    return modelo     


@app.route('/estadisticaModelo/<name>')
def estadisticaModelo(name):  
    global modelo
    global nombreDataset
    
    resultado = modelo.copy()

    respuesta={}
    nombresplit = nombreDataset.replace("dataset_", "")                      

    if(name == 'xgboost-Smote'):   
        respuesta = resultado[nombresplit+'_SMOTE']['XGBoost']
    elif(name == 'xgboost-Smote-Borderline'):  
        respuesta = resultado[nombresplit+'_Borderline']['XGBoost']
    elif(name == 'xgboost-SmoteNC'):   
        respuesta = resultado[nombresplit+'_NC']['XGBoost']
    elif(name == 'xgboost-Random under-sampling'):
        respuesta = resultado[nombresplit+'_RandomUnder']['XGBoost']
    elif(name == 'KNN-Smote'):   
        respuesta = resultado[nombresplit+'_SMOTE']['KNN']
    elif(name == 'KNN-Smote-Borderline'): 
        respuesta = resultado[nombresplit+'_Borderline']['KNN']
    elif(name == 'KNN-SmoteNC'):   
        respuesta = resultado[nombresplit+'_NC']['KNN']
    elif(name == 'KNN-Random under-sampling'):
        respuesta = resultado[nombresplit+'_RandomUnder']['KNN']
    elif(name == 'xgboost-datos_crudos'):
        respuesta = resultado['datos_crudos']['XGBoost']
    elif(name == 'KNN-datos_crudos'):
        respuesta = resultado['datos_crudos']['KNN']

    return respuesta    

@app.route('/descargarModelo/<filename>')
def descargarModelo(filename):

    directory = Imports.os.path.join(app.config['UPLOAD_FOLDER'], filename+'.csv')

    return send_from_directory(directory=Imports.os.path.dirname(directory),
                                   path=Imports.os.path.basename(directory),
                                   as_attachment=True)

@app.route('/cargarCurva/<name>/<metodo>')
def cargarCurva(name,metodo):
    global dependiente
    global nombreDataset

    target = dependiente

    if(name == 'Smote'):   
        respuesta = Imports.pd.read_csv(Imports.os.path.join(app.config['UPLOAD_FOLDER'], nombreDataset + '_smote.csv'))
    elif(name == 'Smote-Borderline'):  
        respuesta = Imports.pd.read_csv(Imports.os.path.join(app.config['UPLOAD_FOLDER'],nombreDataset + '_smote_borderline.csv'))
    elif(name == 'SmoteNC'):   
        respuesta = Imports.pd.read_csv(Imports.os.path.join(app.config['UPLOAD_FOLDER'],nombreDataset + '_smote_nc.csv'))
    elif(name == 'Random under-sampling'):
        respuesta = Imports.pd.read_csv(Imports.os.path.join(app.config['UPLOAD_FOLDER'],nombreDataset + '_random_under.csv'))
    elif(name == 'datos_crudos'):
        respuesta = Imports.pd.read_csv(Imports.os.path.join(app.config['UPLOAD_FOLDER'],'base_test.csv'))
    return imagenesModelo(respuesta, target, metodo, name)


@app.route('/verDistribucion/<name>')
def verDistribucion(name):
    if name == 'smote':
        imagenes = [
            "/static/uploads/Datos_crudos_proporcion.png",
            "/static/uploads/smote_proporcion.png"
        ]
    elif name == 'smote_borderline':
        imagenes = [
            "/static/uploads/Datos_crudos_proporcion.png",
            "/static/uploads/smote_borderline_proporcion.png"
        ]
    elif name == 'smote_nc':
        imagenes = [
            "/static/uploads/Datos_crudos_proporcion.png",
            "/static/uploads/smote_nc_proporcion.png"
        ]
    elif name == 'random_under':
        imagenes = [
            "/static/uploads/Datos_crudos_proporcion.png",
            "/static/uploads/random_under_proporcion.png"
        ]
    else:
        imagenes = [
            "/static/uploads/Datos_crudos_proporcion.png"
        ]

    return jsonify(imagenes)



@app.route('/estadisticaModeloGanador')
def estadisticaModeloGanador():
    global modelo
    global nombreDataset

    copia = modelo.copy()
    nombresplit = nombreDataset.replace("dataset_", "")
    resultados_renombrados = {}

    for metodo, modelos in copia.items():
        # Extraer el nombre del método de balanceo
        if metodo.lower().startswith(nombresplit+"_"):
            metodo_balanceo = metodo.split('_', 1)[-1]
        else:
            metodo_balanceo = metodo

        # Manejo especial para 'base'
        metodo_balanceo_formateado = metodo_balanceo.capitalize() if metodo_balanceo != 'base' else 'Dataset Crudo'
        if metodo_balanceo_formateado == 'Borderline':
            metodo_balanceo_formateado = 'Smote Borderline'
        elif  metodo_balanceo_formateado == 'Nc': 
            metodo_balanceo_formateado = 'SmoteNC'
        elif  metodo_balanceo_formateado == 'Randomunder': 
            metodo_balanceo_formateado = 'Random under-sampling'

        for modelo1, metricas in modelos.items():
            clave_nueva = f"{modelo1}-{metodo_balanceo_formateado}"
            resultados_renombrados[clave_nueva] = metricas

    return jsonify(resultados_renombrados)

if __name__ == '__main__':
    app.run(debug=True)


    
