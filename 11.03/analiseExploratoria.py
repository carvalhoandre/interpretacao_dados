import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import seaborn as sns
import warnings
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import shap

df = pd.read_csv(r'D:\Users\andre\Documents\GitHub\interpretacao_dados\11.03\docs\train.csv')

df.describe()

df.international_plan.value_counts
df.churn.value_counts().plot(kind='pie')
df.international_plan.value_counts().plot(kind='pie')
print('\n================================')

churn_y = df.loc[df.churn == 'yes', 'churn'].count()
churn_n = df.loc[df.churn == 'no', 'churn'].count()
churn_total = churn_y+churn_n
print('Quantidade total de clientes')
print('Quantidade total de clientes com churn ' + str(churn_y) + ' que repesenta ' + str(round(100*churn_y/churn_total, 0)) + '%da base cleintes')
print('Quantidade de clientes sem churn ' +str(churn_n) + ' que representa ' + str(round(100*churn_n/churn_total,0)) + '% da base de clientes')

df.isnull().sum()  # Checa dados ausentes na base

df.state.value_counts()

df.state.value_counts().plot(kind='bar')
print('Grafico bar')

df.number_customer_service_calls.value_counts()

sns.barplot(x=df.voice_mail_plan, y=df.number_vmail_messages, estimator=sum)

df.number_customer_service_calls.value_counts().plot(kind='barh')
print('Grfico barh')

fig, ax = plt.subplots(figsize=(20,5))
sns.countplot(data = df, x='state', order=df['state'].value_counts().index, palette='viridis', hue='churn')
plt.xticks(rotation=90)
plt.xlabel('State', fontsize=10, fontweight='bold')
plt.ylabel('Customers', fontsize=10, fontweight='bold')
plt.title('Estados dos clientes com e sem churn', fontsize=12, fontweight='bold')
plt.show()

# Correlação entre as variáveis 
print('==='*18)
print('Correlação entre as variáveis ')
corr = df.corr()
fig2, ax = plt.subplots(figsize=(15,7))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True, cmap='YlGnBu', annot_kws={'size': 12}, fmt='.2f')

df['churn'] = df['churn'].astype('category')
df['churn'] = df['churn'].cat.codes

df['voice_mail_plan'] = df['voice_mail_plan'].astype('category')
df['voice_mail_plan'] = df['voice_mail_plan'].cat.codes

df['international_plan'] = df['international_plan'].astype('category')
df['international_plan'] = df['international_plan'].cat.codes

df = pd.get_dummies(df, columns=['area_code'], drop_first=True)

df['state'] = df['state'].astype('category')
df['state'] = df['state'].cat.codes

df.sample(4)

# Separando variáveis independentes e dependente
X = df.drop('churn', axis=1)
y = df['churn']

# Dividindo em treino e teste usando train_test_split, 30% de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=987)

xgb_model = XGBClassifier(max_depth=10, 
                          learning_rate=0.1,
                          n_estimators=300, 
                          silent=True,
                          objective='binary:logistic',
                          nthread=-1,
                          missing=np.nan,
                          subsample=0.9, 
                          colsample_bytree=0.9, 
                          base_score=0.5, 
                          seed=142)

xgb_model.fit(X_train, y_train)

prob = xgb_model.predict_proba(X_test)[:,1]
pred = xgb_model.predict(X_test)

print('AUC: '+str(metrics.roc_auc_score(y_test, prob)))
print('Accuracy : '+str(metrics.accuracy_score(y_test, pred)))
print('Recall : '+str(metrics.recall_score(y_test, pred)))
print('F1-Measure : '+str(metrics.f1_score(y_test, pred)))

cm = confusion_matrix(y_test, xgb_model.predict(X_test))
sns.heatmap(cm)

# Feature Importance 
feature_important = xgb_model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=['score']).sort_values(by = 'score', ascending=True)
data.nlargest(20, columns='score').plot(kind='barh', figsize = (20,10))

select_feature = SelectKBest(chi2, k=5).fit(X_test, y_test)
a = select_feature.scores_
b = X_train.columns
df2 = pd.DataFrame(list(zip(b, a)), columns=['Column', 'Score'])

df2['Score'] = df2['Score'].replace(np.nan, 0)
df2['Score'] = df2['Score'].astype(int)
df2.sort_values(by='Score', ascending=False)

#shap
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

shap.plots.waterfall(shap_values[3])


shap.initjs()
shap.plots.force(shap_values[1]) # [1] é o indice da base de teste (X_test)

shap.initjs()
shap.plots.force(shap_values)
 
shap.summary_plot(shap_values)

shap.plots.bar(shap_values)

shap.plots.scatter(shap_values[:,'total_day_minutes'], color=shap_values)

explainer = shap.TreeExplainer(xgb_model)
expected_value = explainer.expected_value
if isinstance(expected_value, list):
    expected_value = expected_value[1]
print(f"Explainer expected value: {expected_value}")

select = range(20)
features = X_test.iloc[select]
features_display = X_test.loc[features.index]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(features)[1]
    shap_interaction_values = explainer.shap_interaction_values(features)
if isinstance(shap_interaction_values, list):
    shap_interaction_values = shap_interaction_values[1]
    
shap.decision_plot(expected_value, shap_values, features_display)

shap.plots.scatter(shap_values[:,'total_day_minutes'], color=shap_values)

