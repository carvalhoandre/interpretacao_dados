import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
# Bibliotecas para o modelo e feature engineering
from pandas_profiling import ProfileReport
# Bibliotecas para o modelo e feature engineering
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (roc_curve,
                             recall_score,
                             accuracy_score,
                             precision_score,
                             f1_score)

from sklearn.ensemble import (AdaBoostClassifier,
                              GradientBoostingClassifier,
                              ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv(r'D:\Users\andre\Documents\Faculdade\inteligencia artificial\18.02\document\aug_train.csv', sep=',')
df_test = pd.read_csv(r'D:\Users\andre\Documents\Faculdade\inteligencia artificial\18.02\document\aug_test.csv', sep=',')

# Checa a quantidade de registros e colunas
df_train.shape, df_test.shape

df_train.sample(5)

# Renomeando as colunas
df_train.columns = ['id', 'genero', 'idade', 'cnh', 'cod_regiao', 'segurado_anteriormente', 'idade_veiculo',
                    'danos_veiculo', 'premio_anual', 'politica_canal_vendas', 'dias_segurado', 'target']

# Gerando report do dataset com pandas profiling
profile = ProfileReport(df_train, title="Pandas Profiling Report", explorative=True)

profile.to_widgets()

profile.to_file(r'D:\Users\andre\Documents\Faculdade\inteligenciaArtificial\18.02\report.html')

# Informações do dataset
df_train.describe()

df_train.sample(4)



# Label encoder para transformação de variável categorica
le = preprocessing.LabelEncoder()
df_train['genero'] = le.fit_transform(df_train['genero'])

# Aplica categorias numerica em dado categorico de texto, mesma forma do Label Encoder mas, mais simples.
df_train['danos_veiculo'] = df_train['danos_veiculo'].astype('category')
df_train['danos_veiculo'] = df_train['danos_veiculo'].cat.codes

# Get dummies para transformar a variável categorica em varias variáveis binárias
df_train = pd.get_dummies(df_train, columns=['idade_veiculo'], drop_first=True)

# X recebe as variáveis independentes, e excluimos a variável Target. O y recebe apenas a variável target.
X = df_train.drop('target', axis=1)
y = df_train['target']

# Aqui dividimos entre treino e teste com 30% em teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7565)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


class ModeloAuxiliar(object):
    def __init__(self, clf, seed=123, params=None):
        if params:
            params['random_state'] = seed
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_

    def score(self, x, y):
        return self.clf.score(x, y)

    class ModeloAuxiliar(object):
        def __init__(self, clf, seed=123, params=None):
            if params:
                params['random_state'] = seed
                self.clf = clf(**params)
            else:
                self.clf = clf()

        def predict(self, x):
            return self.clf.predict(x)

        def fit(self, x, y):
            return self.clf.fit(x, y)

        def feature_importances(self, x, y):
            return self.clf.fit(x, y).feature_importances_

        def score(self, x, y):
            return self.clf.score(x, y)

        # Cria um dicionário (resultados) que armazena o modelo, predição na base de teste,
        # Acurária, e predição. A idéia é transformar em um dataframe para avaliaçãod e
        # qual modelo performa melhor
        resultados = []
        for model in modelos:
            x = ModeloAuxiliar(clf=model['modelo'])
            # treinar o modelo
            x.fit(X_train, y_train)

            # gerar predicao
            x_pred = x.predict(X_test)

            # gerar score
            acuracidade = round(x.score(X_test, y_test) * 100, 2)

            resultados.append({'nome': model['nome'],
                               'score': acuracidade,
                               'pred': x_pred})

            # Transforma o dicionário resultados em um dataframe e armazena no dataframe chamado models
            models = pd.DataFrame(resultados)

            # Apresenta o nome e score dos modelos ordenando pelo score
            models[['nome', 'score']].sort_values(by='score', ascending=False)
