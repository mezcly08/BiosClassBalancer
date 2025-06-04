import os
import pandas as pd  # Importar pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import json
import plotly.express as px

# Scikit-learn
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, recall_score, confusion_matrix, matthews_corrcoef, roc_curve, auc, balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor

# Flask-WTF
from wtforms.fields import StringField, PasswordField, SubmitField

# Imbalanced-learn
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import BorderlineSMOTE, SMOTENC, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

#Balanceo datos
from collections import Counter
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, wasserstein_distance



class Imports:
    """Clase que centraliza las importaciones para facilitar su uso en múltiples módulos."""
    io = io
    os = os
    pd = pd
    np = np
    plt = plt
    sns = sns
    base64 = base64
    json = json
    Counter = Counter
    ks_2samp = ks_2samp
    wasserstein_distance = wasserstein_distance
    px = px
    
    # Módulos de scikit-learn
    KNNImputer = KNNImputer
    OrdinalEncoder = OrdinalEncoder
    LinearRegression = LinearRegression
    RandomForestRegressor = RandomForestRegressor
    RandomForestClassifier = RandomForestClassifier
    mean_squared_error = mean_squared_error
    accuracy_score = accuracy_score
    KMeans = KMeans
    f1_score = f1_score
    recall_score = recall_score
    confusion_matrix = confusion_matrix
    matthews_corrcoef = matthews_corrcoef
    roc_curve = roc_curve
    auc = auc
    balanced_accuracy_score = balanced_accuracy_score
    ConfusionMatrixDisplay = ConfusionMatrixDisplay

    # Módulos de Flask-WTF
    StringField = StringField
    PasswordField = PasswordField
    SubmitField = SubmitField

    # Imbalanced-learn
    make_pipeline = make_pipeline
    BorderlineSMOTE = BorderlineSMOTE
    SMOTENC = SMOTENC
    SMOTE = SMOTE
    RandomUnderSampler = RandomUnderSampler

    #metodos
    XGBRegressor = XGBRegressor
    XGBClassifier = XGBClassifier
    LGBMClassifier = LGBMClassifier
    train_test_split = train_test_split
    KNeighborsRegressor = KNeighborsRegressor
    SimpleImputer = SimpleImputer
    KNeighborsClassifier = KNeighborsClassifier
