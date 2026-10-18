from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

def obtener_datos():
    ds=load_breast_cancer(as_frame=True)
    X=ds.data.copy(); y=ds.target.copy()
    return X,y,ds

def dividir(X,y):
    return train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

def preprocesador(escalar=False):
    pasos=[('imputar',SimpleImputer(strategy='median'))]
    if escalar: pasos.append(('escalar',StandardScaler()))
    return Pipeline(pasos)
