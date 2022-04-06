# """ https://colab.research.google.com/drive/1Cn82mrUqNASIxkXaL-Y1ILg5yqgp2Kdr?usp=sharing """

# Carregando bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import seaborn as sns
import warnings
import pickle as pkl

df = pd.read_csv(r'D:\Users\andre\Documents\GitHub\interpretacao_dados\17.03\document\train.csv')

# Feature Engineering
# Transforma variável churn de sim e não para 1 e 0
df['churn'] = df['churn'].astype('category')
df['churn'] = df['churn'].cat.codes

# Transforma voice_mail_plan de sim e não para 1 e 0
df['voice_mail_plan'] = df['voice_mail_plan'].astype('category')
df['voice_mail_plan'] = df['voice_mail_plan'].cat.codes

# Transforma voice_mail_plan de sim e não para 1 e 0
df['international_plan'] = df['international_plan'].astype('category')
df['international_plan'] = df['international_plan'].cat.codes

# Transforma area_code em numerica
df['area_code'] = df['area_code'].astype('category')
df['area_code'] = df['area_code'].cat.codes

# Transforma state em numerica
df['state'] = df['state'].astype('category')
df['state'] = df['state'].cat.codes

# Separa os datasets em treino e teste
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1111)

# Modelo XGBoost com parametros
xgb_model = XGBClassifier(max_depth=10,
                          learning_rate=0.1,
                          n_estimators=300,
                          silent=True,
                          objective="binary:logistic",
                          nthread=-1,
                          missing=np.nan,
                          subsample=0.9,
                          colsample_bytree=0.9,
                          base_score=0.5,
                          seed=142)

xgb_model.fit(X_train, y_train)

prob = xgb_model.predict_proba(X_test)[:, 1]
pred = xgb_model.predict(X_test)

print("AUC: "+str(metrics.roc_auc_score(y_test, prob)))
print("Accuracy: "+str(metrics.accuracy_score(y_test, pred)))
print("Recall: "+str(metrics.recall_score(y_test, pred)))
print("F1-Measure: "+str(metrics.f1_score(y_test, pred)))

# # Salvando modelo para deploy
# with open('xgb_model', 'wb') as files:
#     pkl.dump(xgb_model, files)
