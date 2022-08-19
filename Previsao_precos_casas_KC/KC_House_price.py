import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sn
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import shapiro
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison

df = pd.read_csv('kc_house_data.csv')

#--------------------------------------------------------------------------------------

def yesno(data):
  aux = data.apply(lambda x: 'yes' if x > 0 else 'no')
  return(aux)

def agrupamento(data, col):
  aux = data[['price', col]].groupby(col).median().reset_index()  
  return(aux)

def correlacao(data,variavel):
  ziptest = data[data.zipcode == variavel]
  correlation = ziptest.corr()
  data = sn.heatmap(correlation, annot = True, fmt = ".1f", linewidths = .10)

def boxplot(col1):
  fig = px.box(df, x = col1, y = 'price')
  fig.show()

def hist(data, x):  
  fig = px.histogram(data, x=x, color="level_value", template = 'plotly_dark', barmode = 'group', color_discrete_sequence=['#00688B', '#DAA520'])
  fig.update_layout(
    paper_bgcolor = '#242424',
    plot_bgcolor = '#242424',
    autosize = True)
  fig.show()

#--------------------------------------------------------------------------------------

df.dtypes
df.isnull().sum()
df.describe()
df['date'] = pd.to_datetime(df['date'])
df['zipcode'] = df['zipcode'].astype(str)
df = df.drop(['sqft_living15', 'sqft_lot15'], axis = 1 )

df['Basement?'] = yesno(df['sqft_basement'])
df['water_view'] = yesno(df['waterfront'])
df['Renoveted?'] = yesno(df['yr_renovated'])

for i in range(len(df)):
  if df.loc[i,'bedrooms'] < 1 or df.loc[i,'bathrooms'] < 1:
    df = df.drop(i)
df = df.drop(15870).reset_index()	  	  
df = df.drop(['index'], axis = 1)

df_bath = agrupamento(df,'bathrooms')
df_grade = agrupamento(df,'grade')

df_zipcode = df.groupby(['zipcode']).median().reset_index()
df_zipcode = df_zipcode.sort_values(by = ['price'])
region = df_zipcode['zipcode'].unique()

for i in region:
  lista = df[df['zipcode']==i].index.tolist()
  num = df[df.zipcode == i].loc[:,'price'].median()
  for j in lista:  
     df.loc[j, 'level_value'] = 'Alto' if  df[df.zipcode == i].loc[j,'price'] > num else 'Baixo'

df['mes_ano'] = df['date'].dt.strftime('%Y-%m')
by_date = df[['price', 'mes_ano']].groupby('mes_ano').median().reset_index()
by_date_Baixo = df[df.level_value == 'Baixo'][['price', 'mes_ano']].groupby('mes_ano').median().reset_index()
by_date_Alto = df[df.level_value == 'Alto'][['price', 'mes_ano']].groupby('mes_ano').median().reset_index()

by_yrbuilt_median = df[['price', 'yr_built']].groupby('yr_built').median().reset_index()
by_yrbuilt_Alto = df[df.level_value == 'Alto'][['price', 'yr_built']].groupby('yr_built').median().reset_index()
by_yrbuilt_Baixo = df[df.level_value == 'Baixo'][['price', 'yr_built']].groupby('yr_built').median().reset_index()

df['sazonal'] = df['mes_ano'].apply(lambda x: 'prima-verão' if (x < '2014-08') or (x >  '2015-03') else
                                              'Out-inver')

df['Quartis_price'] = df['price'].apply(lambda x: 1 if x <= 323000 else
                                                2 if (x > 323000) and (x <= 450000) else
                                                3 if (x > 450000) and (x <= 645000) else
                                                4 if (x>645000) and (x < 1127500) else 5)  


#--------------------------------------------------------------------------------------

figura = plt.figure(figsize=(20,20))
sn.heatmap(df.corr(), cmap="YlGnBu", annot=True)

boxplot('grade')
boxplot('condition')
boxplot('bathrooms')
boxplot('view')  
boxplot('floors')
boxplot('bedrooms') 
fig = px.box(df, x = 'price', template = 'plotly_dark', labels = {'price' : 'Preço'}, color_discrete_sequence=['#e1cc55', '#00224e'])
fig.update_layout(
    paper_bgcolor = '#242424',
    plot_bgcolor = '#242424',
    autosize = True
  )
fig.show()


mapa_oficial = px.scatter_mapbox(df, lat='lat',lon='long', hover_name = 'price', 
                                 color = 'Quartis_price', 
                                 labels = {'Quartis_price' : 'Níveis de preço'}, 
                                 title = 'Mapa com todas as casas, dividido por cores que variam do nivel 1 ao 5',
                                 template = 'plotly_dark',
                                 
                                 color_continuous_scale=px.colors.sequential.Cividis_r,
                                 size_max=10,zoom=9)
mapa_oficial.update_layout(mapbox_style = 'carto-darkmatter')
mapa_oficial.update_layout(height = 700, width = 900, margin = {'r':0, 't':45, 'l':0, 'b':0})
mapa_oficial.show() 

fig = px.bar(df_bath, x="bathrooms", 
             y='price', 
             template = 'plotly_dark',
             color_discrete_sequence=['#e1cc55', '#00224e'], 
             labels={
                     'price': 'Preço das casas',
                     'bathrooms': 'Banheiros'}, 
             title ='Preço Mediano das casas de acordo com o número de Banheiros'
 )
fig.update_layout(
    paper_bgcolor = '#242424',
    plot_bgcolor = '#242424',
    autosize = True)
fig.show()

fig = px.bar(df_grade, x="grade", 
             y='price', 
             template = 'plotly_dark',
             color_discrete_sequence=['#e1cc55', '#00224e'], 
             labels={
                     'price': 'Preço das casas',
                      'grade': 'avaliação'}, 
             title ='Preço Mediano das casas de acordo com o nivel da avaliação'
 )
fig.update_layout(
    paper_bgcolor = '#242424',
    plot_bgcolor = '#242424',
    autosize = True)
fig.show()


fig = px.scatter(df, x="sqft_living", y='price', template = 'plotly_dark',labels={
    'price': 'Preço das casas',
    'sqft_living': 'Tamanho interno por metro quadrado'},
    color_discrete_sequence=['#e1cc55', '#00224e'],
    title ='Preço Mediano das casas de acordo com a área interna das casas'
 )
fig.update_layout(
    paper_bgcolor = '#242424',
    plot_bgcolor = '#242424',
    autosize = True)
fig.show()

hist(df,'grade')
hist(df,'bathrooms')  
hist(df,'bedrooms') 
hist(df,'condition') 
hist(df,'waterfront')
hist(df,'view') 
hist(df,'Renoveted?')
hist(df,'Basement?')
 
fig = px.histogram(df, x='grade', color="level_value", 
                   template = 'plotly_dark', 
                   barmode = 'group', 
                   color_discrete_sequence=['#e1cc55', '#123570'], 
                   labels = {'grade' : 'Avaliação', 'level_value': 'valor da casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o nivel da avaliação ')
fig.update_layout(
    paper_bgcolor = '#242424',
    plot_bgcolor = '#242424',
    width = 900,
    autosize = True)
fig.show()

fig = px.histogram(df, x='bathrooms', color="level_value", 
                   template = 'plotly_dark', 
                   barmode = 'group', 
                   color_discrete_sequence=['#e1cc55', '#123570'], 
                   labels = {'bathrooms' : 'Número de Banheiros', 'level_value': 'Valor da Casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o número de banheiros')
fig.update_layout(
    paper_bgcolor = '#242424',
    plot_bgcolor = '#242424',
    width = 900,
    autosize = True)
fig.show()

fig = px.histogram(df, x='bedrooms', color="level_value", 
                   template = 'plotly_dark', 
                   barmode = 'group', 
                   color_discrete_sequence=['#e1cc55', '#123570'], 
                   labels = {'bedrooms' : 'Número de quartos', 'level_value': 'Valor da Casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o número de Quartos')
fig.update_layout(
    paper_bgcolor = '#242424',
    plot_bgcolor = '#242424',
    width = 900,
    autosize = True)
fig.show()

fig = px.histogram(df, x='view', color="level_value", 
                   template = 'plotly_dark', 
                   barmode = 'group', 
                   color_discrete_sequence=['#e1cc55', '#123570'], 
                   labels = {'view' : 'Nível da Vista', 'level_value': 'Valor da Casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o nível da vista da residência')
fig.update_layout(
    paper_bgcolor = '#242424',
    plot_bgcolor = '#242424',
    width = 900,
    autosize = True)
fig.show()

fig = px.line(by_date_Baixo, x="mes_ano", y='price', title='Variação de preço das casas abaixo da média em suas regiões, durante o periodo de Maio de 2014 até maio de 2015')
fig.show()
fig = px.line(by_date_Alto, x="mes_ano", y='price', title='Variação de preço das casas acima da média em suas regiões, durante o periodo de Maio de 2014 até maio de 2015')
fig.show()

fig = px.line(by_yrbuilt_median, x="yr_built", y='price', title='Variação de preço de todas as casas, de acordo com seu ano de contrução')
fig.show()
fig = px.line(by_yrbuilt_Alto, x="yr_built", y='price', title='Variação de preço de todas as casas acima da mediana de suas regiões, de acordo com seu ano de contrução')
fig.show()
fig = px.line(by_yrbuilt_Baixo, x="yr_built", y='price', title='Variação de preço de todas as casas abaixo da mediana de suas regiões, de acordo com seu ano de contrução')
fig.show()

fig = px.line(by_date, x="mes_ano", 
              y='price',
              color_discrete_sequence=['#e1cc55', '#123570'], 
              template = 'plotly_dark',
              labels = {'price': 'Preço das Casas'},
              title='Variação de preço das casas abaixo da média em suas regiões, durante o periodo de Maio de 2014 até maio de 2015')
fig.update_layout(
    paper_bgcolor = '#242424',
    plot_bgcolor = '#242424',
    width = 1100,
    autosize = True)
fig.show()

#--------------------------------------------------------------------------------------

df = df.drop(['lat'], axis = 1)
df = df.drop(['long'], axis = 1)
df = df.drop(['sqft_above'], axis = 1)
df = df.drop(['sqft_basement'], axis = 1)
df = df.drop(['level_value'], axis = 1)
df = df.drop(['sqft_lot'], axis = 1)
df = df.drop(['waterfront'], axis = 1)
df = df.drop(['Quartis_price'], axis = 1)


label_encoder_zipcode = LabelEncoder()
label_encoder_Renovated = LabelEncoder()
label_encoder_Basement = LabelEncoder()
label_encoder_mes_ano = LabelEncoder()
label_encoder_sazonal = LabelEncoder()
label_encoder_waterfront = LabelEncoder()

x_kc = df.iloc[:, 3:18].values
y_kc = df.iloc[:, 2].values

x_kc[:,9] = label_encoder_zipcode.fit_transform(x_kc[:,9])
x_kc[:,10] = label_encoder_Basement.fit_transform(x_kc[:,10])
x_kc[:,11] = label_encoder_waterfront.fit_transform(x_kc[:,11])
x_kc[:,12] = label_encoder_Renovated.fit_transform(x_kc[:,12])
x_kc[:,13] = label_encoder_mes_ano.fit_transform(x_kc[:,13])
x_kc[:,14] = label_encoder_sazonal.fit_transform(x_kc[:,14])

OneHotEncoder_hr = ColumnTransformer(transformers=[('Onehot', OneHotEncoder(), [9,10,11,12,13,14])], remainder = 'passthrough')
x_kc = OneHotEncoder_hr.fit_transform(x_kc)
x_kc = x_kc.toarray()

x_kc_treinamento, x_kc_teste, y_kc_treinamento, y_kc_teste = train_test_split(x_kc, y_kc, test_size = 0.15, random_state = 0)

scaler_x_kc = StandardScaler()
x_kc_treinamento_scaled = scaler_x_kc.fit_transform(x_kc_treinamento)
scaler_y_kc = StandardScaler()
y_kc_treinamento_scaled = scaler_y_kc.fit_transform(y_kc_treinamento.reshape(-1,1))

x_kc_teste_scaled = scaler_x_kc.transform(x_kc_teste)
y_kc_teste_scaled = scaler_y_kc.transform(y_kc_teste.reshape(-1,1))

regressor_multiplo_kc = LinearRegression()
regressor_multiplo_kc.fit(x_kc_treinamento, y_kc_treinamento)
regressor_multiplo_kc.score(x_kc_treinamento, y_kc_treinamento)
regressor_multiplo_kc.score(x_kc_teste, y_kc_teste)
previsoes = regressor_multiplo_kc.predict(x_kc_teste)
mean_absolute_error(y_kc_teste, previsoes)

poly = PolynomialFeatures(degree = 2)
x_kc_treinamento_poly = poly.fit_transform(x_kc_treinamento)
x_kc_teste_poly = poly.transform(x_kc_teste)

regressor_kc_poly = LinearRegression()
regressor_kc_poly.fit(x_kc_treinamento_poly, y_kc_treinamento)                      
regressor_kc_poly.score(x_kc_treinamento_poly, y_kc_treinamento)
regressor_kc_poly.score(x_kc_teste_poly, y_kc_teste)                     
previsoes_poly = regressor_kc_poly.predict(x_kc_teste_poly)
previsoes_poly         
mean_absolute_error(y_kc_teste, previsoes_poly)    

regressor_arvore_kc = DecisionTreeRegressor()
regressor_arvore_kc.fit(x_kc_treinamento, y_kc_treinamento)
regressor_arvore_kc.score(x_kc_treinamento, y_kc_treinamento)
regressor_arvore_kc.score(x_kc_teste, y_kc_teste)
previsoes_arvore = regressor_arvore_kc.predict(x_kc_teste)
mean_absolute_error(y_kc_teste, previsoes_arvore)

regressor_random_kc = RandomForestRegressor(n_estimators = 100)
regressor_random_kc.fit(x_kc_treinamento, y_kc_treinamento)
regressor_random_kc.score(x_kc_teste, y_kc_teste)
previsoes_random = regressor_random_kc.predict(x_kc_teste)
mean_absolute_error(y_kc_teste, previsoes_random)

regressor_svr_kc = SVR(kernel='rbf')
regressor_svr_kc.fit(x_kc_treinamento_scaled, y_kc_treinamento_scaled.ravel())
regressor_svr_kc.score(x_kc_teste_scaled, y_kc_teste_scaled)
previsoes = regressor_svr_kc.predict(x_kc_teste_scaled)
previsoes = previsoes.reshape(-1, 1)
y_kc_teste_inverse = scaler_y_kc.inverse_transform(y_kc_teste_scaled)
previsoes_inverse = scaler_y_kc.inverse_transform(previsoes)
mean_absolute_error(y_kc_teste_inverse, previsoes_inverse)

regressor_rna_kc = MLPRegressor(activation= 'relu',batch_size= 56, solver = 'sgd')
regressor_rna_kc.fit(x_kc_treinamento_scaled, y_kc_treinamento_scaled.ravel())
regressor_rna_kc.score(x_kc_teste_scaled, y_kc_teste_scaled)
previsoes = regressor_rna_kc.predict(x_kc_teste_scaled)
previsoes = previsoes.reshape(-1,1)
y_kc_teste_inverse_rna = scaler_y_kc.inverse_transform(y_kc_teste_scaled)
previsoes_inverse_rna = scaler_y_kc.inverse_transform(previsoes)
mean_absolute_error(y_kc_teste_inverse_rna, previsoes_inverse_rna)

#--------------------------------------------------------------------------------------

x_kc_house = np.concatenate((x_kc_treinamento, x_kc_teste), axis = 0)
y_kc_house = np.concatenate((y_kc_treinamento, y_kc_teste), axis = 0)

parametros_random = {'n_estimators': [10,50,100,150],
             'min_samples_split' : [2,5,10],
             'min_samples_leaf' : [1, 5, 10],
             'n_jobs' : [-1]}

grid_search = GridSearchCV(estimator = RandomForestRegressor(), param_grid = parametros_random)
grid_search.fit(x_kc_house, y_kc_house)

melhores_parametros = grid_search.best_params_
melhor_score = grid_search.best_score_
print(melhores_parametros)
melhor_score

x_kc_scaled = np.concatenate((x_kc_treinamento_scaled, x_kc_teste_scaled), axis = 0)
y_kc_scaled = np.concatenate((y_kc_treinamento_scaled, y_kc_teste_scaled), axis = 0)

parametros_svr = {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']}
y_kc_scaled = y_kc_scaled.ravel()

grid_search = GridSearchCV(estimator = SVR(), param_grid = parametros_svr)
grid_search.fit(x_kc_scaled, y_kc_scaled)

melhores_parametros_svr = grid_search.best_params_
melhor_score_svr = grid_search.best_score_
print(melhores_parametros_svr)
melhor_score_svr

parametros_mlp = {'activation' : ['logistic', 'tanh', 'relu'],
                 'solver' : ['adam', 'sgd'],
                 'batch_size' : [10, 56]}

grid_search = GridSearchCV(estimator = MLPRegressor(), param_grid = parametros_mlp)
grid_search.fit(x_kc_scaled, y_kc_scaled)

melhores_parametros_mlp = grid_search.best_params_
melhor_score_mlp = grid_search.best_score_
print(melhores_parametros_mlp)
melhor_score_mlp


resultados_random_forest = []
resultados_svm = []
resultados_rede_neural = []

for i in range(30):
    kfold = KFold(n_splits = 10, shuffle = True, random_state = i)
    random_forest = RandomForestRegressor(min_samples_leaf = 1,
                                          min_samples_split= 5,
                                          n_estimators= 100,
                                          n_jobs = -1)
    scores = cross_val_score(random_forest, x_kc_house, y_kc_house, cv = kfold)
    resultados_random_forest.append(scores.mean())
    
    svm = SVR(kernel = 'poly')   
    scores = cross_val_score(svm, x_kc_scaled, y_kc_scaled, cv = kfold)
    resultados_svm.append(scores.mean())
    
    MLP_Regressor = MLPRegressor(activation= 'relu',
                                 batch_size= 56,
                                 solver = 'sgd')   
    scores = cross_val_score(MLP_Regressor, x_kc_scaled, y_kc_scaled, cv = kfold)
    resultados_rede_neural.append(scores.mean()) 

resultados = pd.DataFrame()
resultados = pd.DataFrame({'Random Forest': resultados_random_forest,
                           'SVM': resultados_svm, 
                           'Rede neural': resultados_rede_neural})
resultados.describe()

shapiro(resultados_random_forest), shapiro(resultados_svm), shapiro(resultados_rede_neural)

_, p = f_oneway(resultados_random_forest, resultados_svm, resultados_rede_neural)

resultados_algoritmos = {'accuracy': np.concatenate([resultados_random_forest,resultados_svm, resultados_rede_neural]),
                         'algoritmo': [
                          'random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest',
                          'random_forest','random_forest','random_forest','random_forest',
                          'svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm',
                          'rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural']}

resultados_df = pd.DataFrame(resultados_algoritmos)
resultados_df

compara_algoritmos = MultiComparison(resultados_df['accuracy'], resultados_df['algoritmo'])
teste_estatistico = compara_algoritmos.tukeyhsd()
print(teste_estatistico)