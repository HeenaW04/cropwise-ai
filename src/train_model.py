"""Train and compare crop recommendation classifiers; all reported metrics are measured."""
from pathlib import Path
import time, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'Crop_recommendation.csv'
OUT=ROOT/'results'; FIG=OUT/'figures'; MODELS=ROOT/'models'
FEATURES=['N','P','K','temperature','humidity','ph','rainfall']; TARGET='label'; SEED=42

def main():
    for p in [OUT,FIG,MODELS]: p.mkdir(parents=True,exist_ok=True)
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA}. In Colab, download Crop_recommendation.csv into the data folder first. See README.md.")
    raw=pd.read_csv(DATA)
    # Normalize column names from common versions of this dataset.
    raw.columns=[str(c).strip() for c in raw.columns]
    rename={c:c.lower() for c in raw.columns}
    raw=raw.rename(columns=rename)
    expected=['n','p','k','temperature','humidity','ph','rainfall','label']
    missing=[c for c in expected if c not in raw.columns]
    if missing: raise ValueError(f"Expected columns {expected}; missing {missing}; found {list(raw.columns)}")
    df=raw[expected].copy()
    for c in expected[:-1]: df[c]=pd.to_numeric(df[c],errors='coerce')
    df['label']=df['label'].astype('string').str.strip().str.lower()
    df=df.replace([np.inf,-np.inf],np.nan)
    before=len(df)
    df=df.dropna(subset=['label'])
    df=df.drop_duplicates()
    # Numeric missing values are imputed inside each model pipeline to avoid leakage.
    X=df[expected[:-1]]; y=df['label'].astype(str)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=SEED,stratify=y)
    # EDA figures
    sns.set_theme()
    plt.figure(figsize=(9,5)); y.value_counts().sort_values().plot(kind='barh'); plt.title('Crop Class Distribution'); plt.xlabel('Records'); plt.tight_layout(); plt.savefig(FIG/'class_distribution.png',dpi=180); plt.close()
    plt.figure(figsize=(9,7)); sns.heatmap(df[expected[:-1]].corr(numeric_only=True),annot=True,cmap='coolwarm',fmt='.2f'); plt.title('Feature Correlation Matrix'); plt.tight_layout(); plt.savefig(FIG/'correlation_heatmap.png',dpi=180); plt.close()
    models={
      'Decision Tree':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',DecisionTreeClassifier(random_state=SEED))]),
      'Random Forest':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=-1))]),
      'SVM (RBF)':Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',SVC(kernel='rbf',C=10,probability=True,random_state=SEED))]),
      'K-Nearest Neighbors':Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',KNeighborsClassifier(n_neighbors=5))]),
      'Logistic Regression':Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',LogisticRegression(max_iter=3000,random_state=SEED))]),
      'Extra Trees':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',ExtraTreesClassifier(n_estimators=300,random_state=SEED,n_jobs=-1))]),
    }
    rows=[]; fitted={}
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    for name,model in models.items():
        t=time.perf_counter(); model.fit(X_train,y_train); fit_seconds=time.perf_counter()-t
        pred=model.predict(X_test)
        # CV is computed on training data only. Pipelines keep scaling/imputation within each fold.
        cv_f1=cross_val_score(model,X_train,y_train,cv=cv,scoring='f1_macro',n_jobs=-1).mean()
        rows.append({'Model':name,'Accuracy':accuracy_score(y_test,pred),'Precision_macro':precision_score(y_test,pred,average='macro',zero_division=0),'Recall_macro':recall_score(y_test,pred,average='macro',zero_division=0),'F1_macro':f1_score(y_test,pred,average='macro',zero_division=0),'CV_F1_macro_mean':cv_f1,'Fit_time_seconds':fit_seconds})
        fitted[name]=model
        print(f"{name}: accuracy={rows[-1]['Accuracy']:.4f}, macro-F1={rows[-1]['F1_macro']:.4f}")
    metrics=pd.DataFrame(rows).sort_values('CV_F1_macro_mean',ascending=False)
    metrics.to_csv(OUT/'model_comparison.csv',index=False)
    best_name=str(metrics.iloc[0]['Model']); best=fitted[best_name]
    joblib.dump(best,MODELS/'best_model.joblib')
    joblib.dump(FEATURES,MODELS/'feature_names.joblib')
    pred=best.predict(X_test)
    pd.DataFrame(classification_report(y_test,pred,output_dict=True,zero_division=0)).transpose().to_csv(OUT/'best_model_classification_report.csv')
    labels=sorted(y.unique()); cm=confusion_matrix(y_test,pred,labels=labels)
    plt.figure(figsize=(14,12)); sns.heatmap(cm,cmap='Blues',xticklabels=labels,yticklabels=labels); plt.title(f'Confusion Matrix — {best_name}'); plt.xlabel('Predicted crop'); plt.ylabel('Actual crop'); plt.xticks(rotation=90); plt.yticks(rotation=0); plt.tight_layout(); plt.savefig(FIG/'confusion_matrix_best_model.png',dpi=180); plt.close()
    plt.figure(figsize=(10,6)); sns.barplot(data=metrics,x='F1_macro',y='Model'); plt.xlim(0,1); plt.title('Model Comparison by Macro F1-score'); plt.tight_layout(); plt.savefig(FIG/'model_f1_comparison.png',dpi=180); plt.close()
    plt.figure(figsize=(10,6)); sns.barplot(data=metrics,x='Accuracy',y='Model'); plt.xlim(0,1); plt.title('Model Comparison by Test Accuracy'); plt.tight_layout(); plt.savefig(FIG/'model_accuracy_comparison.png',dpi=180); plt.close()
    summary={'dataset_rows_original':int(before),'dataset_rows_after_dropna_label_and_duplicates':int(len(df)),'dropped_rows':int(before-len(df)),'features':FEATURES,'target':TARGET,'class_count':int(y.nunique()),'class_names':labels,'train_rows':int(len(X_train)),'test_rows':int(len(X_test)),'random_state':SEED,'test_size':0.2,'best_model_by_macro_f1':best_name,'best_test_accuracy':float(metrics.iloc[0]['Accuracy']),'best_test_macro_f1':float(metrics.iloc[0]['F1_macro'])}
    (OUT/'run_summary.json').write_text(json.dumps(summary,indent=2))
    print('\nMODEL COMPARISON (measured):\n',metrics.to_string(index=False))
    print('\nRun summary:',json.dumps(summary,indent=2))
    print('\nCreated figures in results/figures and trained model in models/.')
if __name__=='__main__': main()
