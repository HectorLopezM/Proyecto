import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np

# Parametro para los graficos:
plot_parameter = False

########################### Exploracion univariante: ###########################

# Abrimos el archivo con los datos
df = pd.read_csv('datos_titanic.csv')

# Exploramos las principales caracteristicas del conjunto de datos
df.head()
df.shape
df.columns.values
df.describe()
df.info()

# Borramos las variables que no parecen discriminativas para la variable objetivo
df = df.drop(['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1)

# Estudiamos el balanceo de la variable objetivo
df.groupby(['Survived']).agg({'Survived': ['count']})/df.shape[0]

# Graficamos las variables numericas

if plot_parameter == True:
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.values
    for column in numeric_columns:
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        fig.suptitle(column, fontsize=16)

        ax.set_title("histograma")
        ax.hist(df[column][~pd.isna(df[column])], bins=25)

        ax2.set_title("gráfico de caja")
        ax2.boxplot(df[column][~pd.isna(df[column])], vert=False)
    plt.show()

# Graficamos las variables categoricas
if plot_parameter == True:
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.values
    for column in categorical_columns:
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(column, fontsize=16)
        df[column][~pd.isna(df[column])].value_counts().plot(kind='bar')
    plt.show()

# Exploramos los missing values
na_columns = pd.isna(df).sum()
print("- Número de missings por variable:\n")
print(na_columns[na_columns>0])
print("\n- Proporción de missings por variable:\n")
print(na_columns[na_columns>0]/df.shape[0])

########################### Data wrangling: ###########################

# Reemplazamos los missing values de la variable edad por la media
print("Variable 'age'")
print("- Media: ", df.Age.mean())
print("- Mediana: ", df.Age.median())
df.loc[df['Age'].isna(), 'Age'] = df['Age'].mean()

########################### Feature engineering: ###########################

# Convertimos las variables categoricas a numericas mediante one-hot-encoding
df = pd.get_dummies(df, columns=['Embarked', 'Sex'])
df.columns.values

# Discretizamos la variable edad
df['d_age'] = pd.cut(df.Age,
                     bins = [0, 18, 40, 65, 90],
                     labels = ['child', 'young', 'adult', 'old'],
                     include_lowest = True)
df['d_age'].head()

df = pd.get_dummies(df, columns=['d_age'])
df = df.drop('Age', axis=1)

# Creamos una variable que indica si el pasajero iba con algún familiar
boolean_with_family = (df.SibSp > 0)|(df.Parch > 0)
df['with_family'] = boolean_with_family.astype('int')
df['with_family'].head()

df.info()

########################### Exploracion multivariante: ###########################

# Calculamos la correlacion entre variables explicativas y entre explicativas con target
import seaborn as sns

if plot_parameter == True:
    corr = df.corr()
    fig, _ = plt.subplots(figsize=(10, 8))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=colormap)

# Graficamos scatterplots entre variables numericas
# Variable numérica
variable_1 = 'Fare'
# Cualquier otra variable
variable_2 = 'Parch'

if plot_parameter == True:
    plt.scatter(df[variable_1], df[variable_2])
    plt.xlabel(variable_1)
    plt.ylabel(variable_2)

########################### Modelizacion 1: clasificacion ###########################

from sklearn.model_selection import train_test_split

X = df.drop(['Survived'], axis=1)
y = df['Survived'].astype('category')

# Partición en train/test con un porcentaje de test del 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Número de registros del dataset de train y de test
print("Número de filas")
print("- Train: ", X_train.shape[0])
print("- Test:  ", X_test.shape[0])

# Carga de los modelos de los módulos correspondientes de Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

# Instanciación vacía de objetos de las clases de modelos
lr = LogisticRegression()
nb = GaussianNB()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()
svc = SVC(probability=True)
gbt = GradientBoostingClassifier()

# Nos creamos un diccionario para almacenar los modelos
models = {'lr': lr,
          'nb':nb,
          'knn': knn,
          'rf': rf,
          'svc': svc,
          'gbt': gbt}
models_names = models.keys()

# Construimos un bucle para iterar sobre los modelos del diccionario e ir utilizando el método de Scikit-learn,
# 'fit(X,y)', para modelizar y en función de X
print("Entrenamiento de los modelos...")
for i, m in enumerate(models_names):
    print(i)
    print("-> " + m)
    models[m].fit(X_train, y_train)

# Creamos un diccionario vacío para almacenar los resultados
predict_probs = {}

# Construimos un bucle para iterar sobre los modelos e ir obteniendo sus predicciones sobre el conjunto de test
# y almacenándolas en el diccionario creado anteriormente
print("Predicciones de los modelos...")
for i,m in enumerate(models_names):
    print("-> " + m)
    # predict_proba devuelve las probabilidades de las dos clases, pero sólo nos interesa la clase 1 ([:,1])
    predict_probs[m] = models[m].predict_proba(X_test)[:, 1]

#  Obtenemos las posiciones de los 1s y de los 0s
index_survived_1 = np.where(y_test == 1)[0]
index_survived_0 = np.where(y_test == 0)[0]

# Dibujamos el grafico de probabilidades de 0s y 1s
if plot_parameter == True:
    for i, m in enumerate(models_names):
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(m)
        plt.hist(predict_probs[m][index_survived_0], bins=40)
        plt.hist(predict_probs[m][index_survived_1], bins=40, alpha=0.5)
        plt.legend(["p10", "p11"])

from sklearn import metrics

# Dibujamos las curvas ROC y precision-recall
if plot_parameter == True:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20.1, 10))
    ax1.set_title("ROC")
    ax1.set_xlabel("1 - Especificidad (FPR)")
    ax1.set_ylabel("Recall (TPR)")
    ax1.grid()

    ax2.set_title("Curva Precision - Recall")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.grid()

    auc_roc = {}
    auc_pr = {}
    for i, m in enumerate(models_names):
        fpr, tpr, _ = metrics.roc_curve(y_test, predict_probs[m])
        auc_roc[m] = np.round(metrics.auc(fpr, tpr), 2)
        ax1.plot(fpr, tpr)

        precision, recall, _ = metrics.precision_recall_curve(y_test, predict_probs[m])
        auc_pr[m] = np.round(metrics.auc(recall, precision), 2)
        ax2.plot(recall, precision)

    ax1.legend(predict_probs.keys(), prop={'size': 15})
    ax1.plot(np.linspace(0, 1, len(fpr)), np.linspace(0, 1, len(fpr)), '--')
    plt.show()

# Seleccionamos el mejor algoritmo bajo el criterio AUC
mejor_modelo_roc = max(auc_roc, key=lambda x: auc_roc[x])
mejor_modelo_pr = max(auc_pr, key=lambda x: auc_pr[x])
print("Mejor modelo según el AUC de la curva ROC:", mejor_modelo_roc, " - ", auc_roc[mejor_modelo_roc])
print("Mejor modelo según el AUC de la curva Precision - Recall:", mejor_modelo_pr, " - ", auc_pr[mejor_modelo_pr])

# Seleccionamos XGBOOST y comprobamos la importancia de las variables:
X.columns.values
gbt.feature_importances_

# Calculamos todas las metricas de evaluacion
mejor_modelo = 'gbt'
precision, recall, umbrales = metrics.precision_recall_curve(y_test, predict_probs[mejor_modelo])
f1score = 2 * precision * recall / (precision + recall)
mejor_f1score = np.max(f1score).round(4) # F1 Score más alto

# Seleccionamos el mejor umbral de probabilidad, que corresponde al que ocupa la posición del F1 Score mayor
mejor_umbral = umbrales[np.argmax(f1score)].round(4)
print("- El mejor umbral (según la métrica F1 Score) es: {} \n- F1 Score asociado {}".format(mejor_umbral, mejor_f1score))

# Calculamos la matriz de confusion
predict_binary = (predict_probs[mejor_modelo] > mejor_umbral).astype(int)
# La función de Scikit-learn nos da la matriz de confusión en la forma real\predicho, pero
# por consistencia con esta sección, la transponemos
matriz_conf = metrics.confusion_matrix(y_test, predict_binary).T
matriz_conf = pd.DataFrame(matriz_conf)

########################### Modelizacion 2: analisis cluster ###########################

from sklearn import cluster
from sklearn.preprocessing import StandardScaler

X = X.drop(['with_family', 'SibSp', 'Embarked_C', 'Embarked_Q', 'Embarked_S'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

np.random.seed(2)
k_means = cluster.KMeans(n_clusters = 2)
k_means.fit(X_scaled)
num_cluster = k_means.labels_

# Informacion sobre los grupos:
print("Se han generado {} grupos".format(k_means.n_clusters))
print("El metodo de inicializacion de los clusters utilizado ha sido: {}".format(k_means.init))
print("La inercia o WCSS de los grupos generados es de {} \n".format(k_means.inertia_))
print("Los centros de los grupos generados son los siguientes: \n {}".format(k_means.cluster_centers_))

# Comparacion entre la variable 'cluster' y la variable objetivo del problema supervisado:
## Comprobamos el porcentaje de observaciones bien clasificadas mediante el analisis cluster
print("Se ha clasificado bien el {}% de las observaciones".format(round(sum(y == num_cluster)/len(y)*100, 2)))

## Comprobamos el porcentaje de cada grupo con respcto al porcentaje de las clases de la variable objetivo
print("-> Porcentaje de observaciones en el grupo 0: {}%".format(round(sum(num_cluster==0)/len(num_cluster)*100, 2)))
print("-> Porcentaje de observaciones en el grupo 1: {}%".format(round(sum(num_cluster==1)/len(num_cluster)*100, 2)))
print("-> Porcentaje de cada clase de la variable objetivo:\n{}".format(df.groupby(['Survived']).agg({'Survived': ['count']})/df.shape[0]))

## Estudiamos la matriz de confusion:
matriz_conf_cluster = metrics.confusion_matrix(y, num_cluster).T
matriz_conf_cluster = pd.DataFrame(matriz_conf_cluster)

# Estudiamos a las observaciones de cada grupo:
## Variables numericas
if plot_parameter == True:
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.values
    for column in numeric_columns:
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(column, fontsize=16)
        plt.scatter(num_cluster, X[column], alpha=0.1)
    plt.show()

## Variables categoricas
X['cluster'] = num_cluster
categorical_columns = X.select_dtypes(include=['int32', 'uint8']).columns.values
np.delete(categorical_columns, len(categorical_columns)-1)

for column in categorical_columns:
    print(X.groupby(['cluster', column]).size())

########################### Modelizacion 3: PCA ###########################

from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA

# Calculamos dos componentes principales:
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

## Creamos el conjunto de datos de componentes principales
X_pca = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])

## Calculamos la varianza explicada por los componentes principales
print('La proporcion de varianza explicada por cada componente es: \n {}'.format(pca.explained_variance_ratio_))
explained_var = round(sum(pca.explained_variance_ratio_)*100, 2)
print('Lo que supone un {}% del total de la varianza del conjunto de datos'.format(explained_var))

## Componentes principales en funcion del cluster
plt.scatter(X_pca.PC1, X_pca.PC2, c = num_cluster)

# Calculamos tres componentes principales:
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_scaled)

## Creamos el conjunto de datos de componentes principales
X_pca = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2', 'PC3'])

## Calculamos la varianza explicada por los componentes principales
print('La proporcion de varianza explicada por cada componente es: \n {}'.format(pca.explained_variance_ratio_))
explained_var = round(sum(pca.explained_variance_ratio_)*100, 2)
print('Lo que supone un {}% del total de la varianza del conjunto de datos'.format(explained_var))

## Componentes principales en funcion del cluster
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X_pca.PC1, X_pca.PC2, X_pca.PC3, c = num_cluster)