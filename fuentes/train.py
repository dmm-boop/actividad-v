import os, json, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from scipy.stats import ttest_rel
from datos_prep import obtener_datos, dividir, preprocesador

ROOT=Path.cwd()
if not (ROOT/'fuentes').exists() and (ROOT.parent/'fuentes').exists(): ROOT=ROOT.parent
(ROOT/'evidencias').mkdir(exist_ok=True)
(ROOT/'datos/datos_ini').mkdir(parents=True,exist_ok=True)
(ROOT/'datos/datos_limp').mkdir(parents=True,exist_ok=True)
X,y,ds=obtener_datos()
raw=X.assign(target=y)
raw.to_csv(ROOT/'datos/datos_ini/breast_cancer_original.csv',index=False)
X_train,X_test,y_train,y_test=dividir(X,y)
X_train.assign(target=y_train).to_csv(ROOT/'datos/datos_limp/train.csv',index=False)
X_test.assign(target=y_test).to_csv(ROOT/'datos/datos_limp/test.csv',index=False)
print('Filas:',len(X),'Variables:',X.shape[1],'Nulos:',int(X.isna().sum().sum()))
print('Clases: 0=maligno, 1=benigno:',y.value_counts().to_dict())
fig,ax=plt.subplots(1,2,figsize=(11,4))
y.value_counts().sort_index().plot.bar(ax=ax[0],title='Distribución de clases')
X[['mean radius','mean texture']].boxplot(ax=ax[1]); ax[1].set_title('Distribuciones y valores extremos')
fig.tight_layout(); fig.savefig(ROOT/'evidencias/eda.png',dpi=160); plt.show()

mlflow.set_tracking_uri((ROOT/'mlruns').resolve().as_uri())
mlflow.set_experiment('Actividad5_BreastCancer')
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
configs={
 'Regresion_logistica':(LogisticRegression(max_iter=2000,random_state=42),True,{'modelo__C':[0.01,0.1,1,10]}),
 'Random_forest':(RandomForestClassifier(random_state=42,n_jobs=1),False,{'modelo__n_estimators':[100,200],'modelo__max_depth':[None,5,10]})
}
results=[]; folds={}; best_models={}
for name,(est,scale,grid) in configs.items():
    pipe=Pipeline([('preparacion',preprocesador(scale)),('modelo',est)])
    with mlflow.start_run(run_name=name):
        search=GridSearchCV(pipe,grid,cv=cv,scoring='roc_auc',n_jobs=1,refit=True)
        search.fit(X_train,y_train)
        best=search.best_estimator_; best_models[name]=best
        scores=cross_validate(best,X_train,y_train,cv=cv,scoring={'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc'},n_jobs=1)
        folds[name]=scores['test_roc_auc']
        proba=best.predict_proba(X_test)[:,1]; pred=best.predict(X_test)
        metrics={'accuracy':accuracy_score(y_test,pred),'precision':precision_score(y_test,pred),'recall':recall_score(y_test,pred),'f1':f1_score(y_test,pred),'roc_auc':roc_auc_score(y_test,proba)}
        mlflow.log_params({**search.best_params_,'dataset':'sklearn_breast_cancer','seed':42,'cv_folds':5,'test_fraction':0.2})
        mlflow.log_metrics({'cv_roc_auc_best':search.best_score_,**{'cv_'+k+'_mean':float(np.mean(v)) for k,v in scores.items() if k.startswith('test_')},**{'test_'+k:v for k,v in metrics.items()}})
        fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay.from_predictions(y_test,pred,ax=ax); fig.tight_layout()
        cm_path=ROOT/f'evidencias/matriz_{name}.png';fig.savefig(cm_path,dpi=160);plt.close(fig)
        mlflow.log_artifact(str(cm_path),'figuras')
        mlflow.sklearn.log_model(best,artifact_path='modelo')
        results.append({'modelo':name,**metrics,'cv_roc_auc':search.best_score_,'parametros':json.dumps(search.best_params_)})
        print(name,metrics,search.best_params_)
res=pd.DataFrame(results);res.to_csv(ROOT/'evidencias/metricas.csv',index=False)
fig,ax=plt.subplots(figsize=(7,4))
res.set_index('modelo')[['accuracy','precision','recall','f1','roc_auc']].T.plot.bar(ax=ax,ylim=(0.75,1.03));ax.legend(loc='lower right');fig.tight_layout();fig.savefig(ROOT/'evidencias/comparacion.png',dpi=160);plt.show()
fig,ax=plt.subplots(figsize=(6,5))
for name,model in best_models.items(): RocCurveDisplay.from_estimator(model,X_test,y_test,ax=ax,name=name)
fig.tight_layout();fig.savefig(ROOT/'evidencias/roc.png',dpi=160);plt.show()
# Comparación exploratoria: folds pareados; reuso de folds tras búsqueda introduce sesgo optimista.
a,b=folds.values();stat,pvalue=ttest_rel(a,b)
(ROOT/'evidencias/estadistica.json').write_text(json.dumps({'prueba':'t pareada exploratoria sobre AUC por fold','t':float(stat),'p_value':float(pvalue),'advertencia':'Los folds se reutilizan después de la selección de hiperparámetros; no es inferencia confirmatoria. Para inferencia robusta usar CV anidada.'},indent=2))
print('t pareada exploratoria:',stat,'p=',pvalue)
print('MLflow:',(ROOT/'mlruns').resolve())
