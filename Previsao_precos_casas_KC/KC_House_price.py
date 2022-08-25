import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sn
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objects as go
from matplotlib import gridspec 

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
import pickle

df = pd.read_csv('kc_house_data.csv')

#--------------------------------------------------------------------------------------

def yesno(data):
  aux = data.apply(lambda x: 'yes' if x > 0 else 'no')
  return(aux)

def agrupamento(data, col):
  aux = data[['price', col]].groupby(col).median().reset_index()  
  return(aux)

def contagem(data, col):
  aux = data[['id', col]].groupby(col).count().reset_index()
  aux = aux.rename(columns={'id': 'Acima'}) 
  # aux['percent'] = (aux['id'] / 
  #                 aux['id'].sum()) * 100 
  # print(aux)                
  return(aux)


def correlacao(data,variavel):
  ziptest = data[data.zipcode == variavel]
  correlation = ziptest.corr()
  data = sn.heatmap(correlation, annot = True, fmt = ".1f", linewidths = .10)

def boxplot(col1):
  fig = px.box(df, x = col1, y = 'price')
  fig.show()

def hist(data, x):  
  fig = px.histogram(data, x=x, color="level_value", template = 'plotly_dark', barmode = 'group', color_discrete_sequence=['#e1cc55', '#00224e'])
  fig.update_layout(
    paper_bgcolor = '#242424',
    plot_bgcolor = '#242424',
    autosize = True)
  fig.show()

def layout(graph):
  graph.update_layout(
      paper_bgcolor = '#242424',
      plot_bgcolor = '#242424',
      autosize = True
    )
  graph.show()
 


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

df['Quartis_price'] = df['price'].apply(lambda x: 1 if x <= 323000 else
                                                2 if (x > 323000) and (x <= 450000) else
                                                3 if (x > 450000) and (x <= 645000) else
                                                4 if (x>645000) and (x < 1127500) else 5)  

df['sqft_level'] = df['sqft_living'].apply(lambda x: 1 if x <= 1430 else 
                                                     2 if (x > 1430) and (x <= 1920) else 
                                                     3 if (x > 1920) and (x <= 2550) else 
                                                     4 if (x > 2550) and (x <= 4230) else 5)

for i in range(len(df)):
  if df.loc[i,'bedrooms'] < 1 or df.loc[i,'bathrooms'] < 1:
    df = df.drop(i)
df = df.drop(15870).reset_index()	  	  
df = df.drop(['index'], axis = 1)

df_zipcode = agrupamento(df,'zipcode')

region = df_zipcode['zipcode'].unique()
for i in region:
  lista = df[df['zipcode']==i].index.tolist()
  num = df[df.zipcode == i].loc[:,'price'].median()
  for j in lista:  
     df.loc[j, 'level_value'] = 'Alto' if  df[df.zipcode == i].loc[j,'price'] > num else 'Baixo'

df_zipcode = df_zipcode.sort_values(by = ['price'])
df_zipcode['price'].mean()
df_zipcode1 = df_zipcode
zipcode_98039 = df[df.zipcode == '98039']
zipcode_98039.describe()
for i in range(len(df_zipcode1)):
  if df_zipcode1.loc[i,'price'] < 501607:
    df_zipcode1 = df_zipcode1.drop(i)
df_zipcode1 = df_zipcode1['zipcode'].values

df['zipcode_class'] = 'abaixo'
for j in df_zipcode1:
  for i in range (len(df)):
    if df.loc[i,'zipcode'] == j:
      df.loc[i,'zipcode_class'] = 'acima'
    
df['zipcode_class'].unique()

df_bath = agrupamento(df,'bathrooms')
df_grade = agrupamento(df,'grade')
df_view = agrupamento(df,'view')
df_bedrooms = agrupamento(df,'bedrooms')
df_condition = agrupamento(df,'condition')

df_high_price = df[df.Quartis_price > 3]

df.columns

df_countbath = contagem(df_high_price,'bathrooms')
df_countbed = contagem(df_high_price,'bedrooms')
df_countgrade = contagem(df_high_price, 'grade')
df_countview = contagem(df_high_price, 'view')
df_countwater = contagem(df_high_price, 'water_view')
df_countcond = contagem(df_high_price, 'condition')

df['level_value'].unique()
df_acima_zipcode = df[df.level_value == 'Alto']
df_baixo_zipcode = df[df.level_value == 'Baixo']


df_countbath1 = contagem(df_acima_zipcode,'bathrooms')
df_countbed1 = contagem(df_acima_zipcode,'bedrooms')
df_countgrade1 = contagem(df_acima_zipcode, 'grade')
df_countview1 = contagem(df_acima_zipcode, 'view')
df_countwater1 = contagem(df_acima_zipcode, 'water_view')
df_countcond1 = contagem(df_acima_zipcode, 'condition')

df_countbath2 = contagem(df_baixo_zipcode,'bathrooms')
df_countbed2 = contagem(df_baixo_zipcode,'bedrooms')
df_countgrade2 = contagem(df_baixo_zipcode, 'grade')
df_countview2 = contagem(df_baixo_zipcode, 'view')
df_countwater2 = contagem(df_baixo_zipcode, 'water_view')
df_countcond2 = contagem(df_baixo_zipcode, 'condition')

# df_high_price.count() #5369
# df_acima_zipcode.count() #10674


df['mes_ano'] = df['date'].dt.strftime('%Y-%m')
by_date = df[['price', 'mes_ano']].groupby('mes_ano').median().reset_index()
by_date_Baixo = df[df.level_value == 'Baixo'][['price', 'mes_ano']].groupby('mes_ano').median().reset_index()
by_date_Alto = df[df.level_value == 'Alto'][['price', 'mes_ano']].groupby('mes_ano').median().reset_index()

by_yrbuilt_median = df[['price', 'yr_built']].groupby('yr_built').median().reset_index()
by_yrbuilt_Alto = df[df.level_value == 'Alto'][['price', 'yr_built']].groupby('yr_built').median().reset_index()
by_yrbuilt_Baixo = df[df.level_value == 'Baixo'][['price', 'yr_built']].groupby('yr_built').median().reset_index()

df['sazonal'] = df['mes_ano'].apply(lambda x: 'prima-verão' if (x < '2014-08') or (x >  '2015-03') else
                                              'Out-inver')
sazonal = df[['price', 'sazonal']].groupby('sazonal').median().reset_index()
# df.to_csv('kc_definitivo.csv', index=False) 

#--------------------------------------------------------------------------------------

figura = plt.figure(figsize=(20,20))
sn.heatmap(df.corr(), cmap="YlGnBu", annot=True)

boxplot('grade')
boxplot('condition')
boxplot('bathrooms')
boxplot('view')  
boxplot('floors')
boxplot('bedrooms') 

fig = px.bar(df_bath, x="bathrooms", 
             y='price', 
             template = 'plotly_dark',
             color_discrete_sequence=['#e1cc55', '#00224e'], 
             labels={
                     'price': 'Preço das casas',
                     'bathrooms': 'Banheiros'}, 
             title ='Preço Mediano das casas de acordo com o número de Banheiros')
layout(fig)
    
hist(df,'grade')
hist(df,'bathrooms')  
hist(df,'bedrooms') 
hist(df,'condition') 
hist(df,'waterfront')
hist(df,'view') 
hist(df,'Renoveted?')
hist(df,'Basement?')
hist(df,'sazonal')

fig = px.bar(sazonal, x="sazonal", 
             y='price', 
             template = 'plotly_dark',
             color_discrete_sequence=['#e1cc55', '#00224e'], 
             labels={
                     'price': 'Preço das casas',
                     'bathrooms': 'Banheiros'}, 
             title ='Preço Mediano das casas de acordo com a sazonalidade', text = 'price')
layout(fig) 

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

fig = px.bar(df_grade, x="grade", 
             y='price', 
             template = 'plotly_dark',
             color_discrete_sequence=['#e1cc55', '#00224e'], 
             labels={
                     'price': 'Preço das casas',
                      'grade': 'avaliação'}, 
             title ='Preço Mediano das casas de acordo com o nivel da avaliação'
 )
layout(fig)

fig = px.bar(df_view, x="view", 
             y='price', 
             template = 'plotly_dark',
             color_discrete_sequence=['#e1cc55', '#00224e'], 
             labels={
                     'price': 'Preço das casas',
                      'view': 'Vista da casa'}, 
             title ='Preço Mediano das casas de acordo com o nivel da vista da casa'
 )
layout(fig)

fig = px.bar(df_bedrooms, x="bedrooms", 
             y='price', 
             template = 'plotly_dark',
             color_discrete_sequence=['#e1cc55', '#00224e'], 
             labels={
                     'price': 'Preço das casas',
                      'bedrooms': 'Número de quartos da casa'}, 
             title ='Preço Mediano das casas de acordo com o Número de quartos da casa'
 )
layout(fig)

fig = px.bar(df_condition, x="condition", 
             y='price', 
             template = 'plotly_dark',
             color_discrete_sequence=['#e1cc55', '#00224e'], 
             labels={
                     'price': 'Preço das casas',
                     'condition': 'Nível da condição'}, 
             title ='Preço Mediano das casas de acordo com o nível da condição da casa'
 )
layout(fig)

fig = px.scatter(df, x="sqft_living", y='price', trendline = 'ols', trendline_color_override = '#892721', template = 'plotly_dark',labels={
    'price': 'Preço das casas',
    'sqft_living': 'Tamanho interno por metro quadrado'},
    color_discrete_sequence=['#e1cc55', '#00224e'],
    title ='Preço Mediano das casas de acordo com a área interna das casas'
 )
layout(fig) 

fig = px.box(df, x = 'price', 
              template = 'plotly_dark', 
              labels = {'price' : 'Preço'}, 
              color_discrete_sequence=['#e1cc55', '#00224e'],
              )
layout(fig)

fig = px.box(df, x = 'sqft_living', 
              template = 'plotly_dark', 
              labels = {'sqft_living' : 'Área interna'}, 
              color_discrete_sequence=['#e1cc55', '#00224e'],
              )
layout(fig)

df_waterfront = agrupamento(df,'water_view')


fig = px.bar(df_waterfront, x="water_view", 
             y='price', 
             template = 'plotly_dark',
             color_discrete_sequence=['#e1cc55', '#00224e'], 
             labels={
                     'price': 'Preço das casas',
                     'water_view': 'Possui vista para o mar'}, 
             title ='Preço Mediano considerando se a casa tem vista para o mar'
 )
layout(fig)


mapa_oficial = px.scatter_mapbox(df, lat='lat',lon='long', hover_name = 'price', 
                                 color = 'Quartis_price', 
                                 labels = {'Quartis_price' : 'Níveis de preço'}, 
                                 title = 'Mapa com todas as casas, dividido por cores que variam do nivel 1 ao 5',
                                 template = 'plotly_dark',
                                 color_continuous_scale=px.colors.sequential.Cividis_r,
                                 size_max=10,zoom=9)
mapa_oficial.update_layout(mapbox_style = 'carto-darkmatter')
mapa_oficial.update_layout(height = 700, width = 750, margin = {'r':0, 't':45, 'l':0, 'b':0})
mapa_oficial.show() 

fig = px.histogram(df, x='grade', color="level_value", 
                   template = 'plotly_dark', 
                   barmode = 'group', 
                   color_discrete_sequence=['#e1cc55', '#123570'], 
                   labels = {'grade' : 'Avaliação', 'level_value': 'valor da casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o nivel da avaliação ')
layout(fig)



fig = px.histogram(df, x='bathrooms', color="level_value", 
                   template = 'plotly_dark', 
                   barmode = 'group',
                   width= 800, 
                   color_discrete_sequence=['#e1cc55', '#123570'], 
                   labels = {'bathrooms' : 'Número de Banheiros', 'level_value': 'Valor da Casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o número de banheiros')
fig.update_traces(textposition='inside',texttemplate='%{text:.2s}')
layout(fig)

fig = px.histogram(df, x='sqft_level', color="level_value", 
                   template = 'plotly_dark', 
                   barmode = 'group', width=760, 
                   color_discrete_sequence=['#e1cc55', '#123570'], 
                   labels = {'sqft_level' : 'Nível do tamanho interno', 'level_value': 'Valor da Casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o tamanho interno')
layout(fig)

fig = px.histogram(df, x='bedrooms', color="level_value", 
                   template = 'plotly_dark', 
                   barmode = 'group', 
                   color_discrete_sequence=['#e1cc55', '#123570'], 
                   labels = {'bedrooms' : 'Número de quartos', 'level_value': 'Valor da Casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o número de Quartos')
layout(fig)

fig = px.histogram(df, x='view', color="level_value", 
                   template = 'plotly_dark', 
                   barmode = 'group',
                   width=750, 
                   color_discrete_sequence=['#e1cc55', '#123570'], 
                   labels = {'view' : 'Nível da Vista', 'level_value': 'Valor da Casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o nível da vista')
layout(fig)

fig = px.line(by_date, x="mes_ano", 
              y='price', width=850,
              color_discrete_sequence=['#e1cc55', '#123570'], 
              template = 'plotly_dark',
              labels = {'price': 'Preço das Casas'},
              title='Variação de preço médio das casas, durante o periodo de Maio de 2014 até maio de 2015')
layout(fig)
#--------------------------------------------------------------------------------------
df = pd.read_csv('kc_definitivo.csv') 


df = df.drop(['lat'], axis = 1)
df = df.drop(['long'], axis = 1)
df = df.drop(['sqft_above'], axis = 1)
df = df.drop(['sqft_basement'], axis = 1)
df = df.drop(['level_value'], axis = 1)
df = df.drop(['sqft_lot'], axis = 1)
df = df.drop(['waterfront'], axis = 1)
df = df.drop(['Quartis_price'], axis = 1)
df= df.drop(['sqft_level'], axis = 1)
df= df.drop(['zipcode_class'], axis = 1)

df['zipcode'] = df['zipcode'].astype(str)


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
regressor_random_kc.score(x_kc_treinamento, y_kc_treinamento)
regressor_random_kc.score(x_kc_teste, y_kc_teste)
previsoes_random = regressor_random_kc.predict(x_kc_teste)
mean_absolute_error(y_kc_teste, previsoes_random)

regressor_svr_kc = SVR(kernel='rbf')
regressor_svr_kc.fit(x_kc_treinamento_scaled, y_kc_treinamento_scaled.ravel())
regressor_svr_kc.score(x_kc_treinamento_scaled, y_kc_treinamento_scaled)
regressor_svr_kc.score(x_kc_teste_scaled, y_kc_teste_scaled)
previsoes = regressor_svr_kc.predict(x_kc_teste_scaled)
previsoes = previsoes.reshape(-1, 1)
y_kc_teste_inverse = scaler_y_kc.inverse_transform(y_kc_teste_scaled)
previsoes_inverse = scaler_y_kc.inverse_transform(previsoes)
mean_absolute_error(y_kc_teste_inverse, previsoes_inverse)

regressor_rna_kc = MLPRegressor()
regressor_rna_kc.fit(x_kc_treinamento_scaled, y_kc_treinamento_scaled.ravel())
regressor_rna_kc.score(x_kc_treinamento_scaled, y_kc_treinamento_scaled)
regressor_rna_kc.score(x_kc_teste_scaled, y_kc_teste_scaled)
previsoes = regressor_rna_kc.predict(x_kc_teste_scaled)
previsoes = previsoes.reshape(-1,1)
y_kc_teste_inverse_rna = scaler_y_kc.inverse_transform(y_kc_teste_scaled)
previsoes_inverse_rna = scaler_y_kc.inverse_transform(previsoes)
mean_absolute_error(y_kc_teste_inverse_rna, previsoes_inverse_rna)

primeiros_resultados = {'Modelos' : ['Regressão Linear', 'Regressão polinominal','Árvore de decisão', 'Random Forest Regressor', 'SVR', 'Rede neural'],
'score treinamento': [0.81223, 0.92051, 0.99997, 0.97751, 0.85804, 0.95287 ],
'score test': [0.80018, -196.89182, 0.62866, 0.79508, 0.79508, 0.85604],
'mean absolute error': [94535.60, 210963.39, 115285.45, 82768.32, 75399.06, 86902.68]}

primeiros_resultados = pd.DataFrame(primeiros_resultados)

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


regressor_svr_kc = SVR(kernel='poly')
regressor_svr_kc.fit(x_kc_treinamento_scaled, y_kc_treinamento_scaled.ravel())
regressor_svr_kc.score(x_kc_treinamento_scaled, y_kc_treinamento_scaled)
regressor_svr_kc.score(x_kc_teste_scaled, y_kc_teste_scaled)
previsoes = regressor_svr_kc.predict(x_kc_teste_scaled)
previsoes = previsoes.reshape(-1, 1)
y_kc_teste_inverse = scaler_y_kc.inverse_transform(y_kc_teste_scaled)
previsoes_inverse = scaler_y_kc.inverse_transform(previsoes)
mean_absolute_error(y_kc_teste_inverse, previsoes_inverse)

regressor_rna_kc = MLPRegressor(activation= 'relu',batch_size= 56, solver = 'sgd',max_iter=1000, hidden_layer_sizes=(9,9))
regressor_rna_kc.fit(x_kc_treinamento_scaled, y_kc_treinamento_scaled.ravel())
regressor_rna_kc.score(x_kc_treinamento_scaled, y_kc_treinamento_scaled)
regressor_rna_kc.score(x_kc_teste_scaled, y_kc_teste_scaled)
previsoes = regressor_rna_kc.predict(x_kc_teste_scaled)
previsoes = previsoes.reshape(-1,1)
y_kc_teste_inverse_rna = scaler_y_kc.inverse_transform(y_kc_teste_scaled)
previsoes_inverse_rna = scaler_y_kc.inverse_transform(previsoes)
mean_absolute_error(y_kc_teste_inverse_rna, previsoes_inverse_rna)

regressor_rna_kc = MLPRegressor(activation= 'relu',batch_size= 56, solver = 'sgd',max_iter=1000, hidden_layer_sizes=(9,9))
regressor_rna_kc.fit(x_kc_scaled, y_kc_scaled.ravel())
regressor_rna_kc.score(x_kc_scaled, y_kc_scaled)
regressor_rna_kc.score(x_kc_teste_scaled, y_kc_teste_scaled)
previsoes = regressor_rna_kc.predict(x_kc_teste_scaled)
previsoes = previsoes.reshape(-1,1)
y_kc_teste_inverse_rna = scaler_y_kc.inverse_transform(y_kc_teste_scaled)
previsoes_inverse_rna = scaler_y_kc.inverse_transform(previsoes)
mean_absolute_error(y_kc_teste_inverse_rna, previsoes_inverse_rna)

pickle.dump(regressor_rna_kc, open ('rede_neural_finalizado.sav', 'wb'))

rede_neural = pickle.load(open('rede_neural_finalizado.sav','rb'))



df.loc[1,'price']

scaler_x_kc = StandardScaler()
x_kc_scaled = scaler_x_kc.fit_transform(x_kc)
scaler_y_kc = StandardScaler()
y_kc_scaled = scaler_y_kc.fit_transform(y_kc.reshape(-1,1))

regressor_rna_kc = MLPRegressor(activation= 'relu',batch_size= 56, solver = 'sgd',max_iter=1000, hidden_layer_sizes=(9,9))
regressor_rna_kc.fit(x_kc_scaled, y_kc_scaled.ravel())
regressor_rna_kc.score(x_kc_scaled, y_kc_scaled)
y_kc_scaled = scaler_y_kc.inverse_transform(y_kc_scaled)

x_kc.shape
x_kc_scaled
df_definitivo= np.concatenate((x_kc_scaled, df.iloc[:,0:3]), axis = 1)

df_definitivo = pd.DataFrame(df_definitivo)

testes = regressor_rna_kc.predict(x_kc_scaled)
testes = testes.reshape(1,-1)
testes = scaler_y_kc.inverse_transform(testes)
testes = testes.ravel()
testes = pd.DataFrame(testes)

df_definitivo[103]= testes

for i in range(len(df_definitivo)):
  if df_definitivo.loc[i,102] < df_definitivo.loc[i,103]:
    df_definitivo.loc[i,104] = 'Abaixo'
  else:
    df_definitivo.loc[i,104] = 'Acima'  

df.iloc[:,17:19]
df_definitivo= np.concatenate((df_definitivo, df.iloc[:,17:19]), axis = 1)

df_definitivo = pd.DataFrame(df_definitivo)

mapa_oficial = px.scatter_mapbox(df_definitivo, lat=105,lon=106,
                                 color = 'valor_casa', hover_name = 'price', 
                                 labels = {'104' : 'Níveis de preço'}, 
                                 title = 'Mapa com todas as casas, dividido por casas abaixo e acima do preço', size= 'previsao', size_max= 15, 
                                 template = 'plotly_dark',
                                 color_discrete_sequence=['#e1cc55', '#123570'],
                                 zoom=9)
mapa_oficial.update_layout(mapbox_style = 'carto-darkmatter')
mapa_oficial.update_layout(height = 700, width = 750, margin = {'r':0, 't':45, 'l':0, 'b':0})
mapa_oficial.show() 

df_definitivo = df_definitivo.rename(columns={100 : 'id',
                        101 : 'data',
                        102 : 'price',
                        103 : 'previsao',
                        104 : 'valor_casa'
                        })

abaixo = df_definitivo[df_definitivo.valor_casa == 'Abaixo']

(abaixo['previsao'] - abaixo['price']).sum() 

df_definitivo.dtypes

df_definitivo['price'] = df_definitivo['price'].astype(float)
df_definitivo['previsao'] = df_definitivo['previsao'].astype(float)

df_zipcode.head(50)

df1 = df
df1_x_kc = df.iloc[:, 3:18]
df1_y_kc = df.iloc[:, 2].values

# df1_x_kc.loc[0,'sqft_living'] = 1950
# df1_x_kc.loc[0,'bathrooms'] = 3
# df1_x_kc.loc[0,'condition'] = 4
# df1_x_kc.loc[0,'grade'] = 8
# df1_x_kc.loc[0,'zipcode'] = '98115'
df1_x_kc.loc[0,'bedrooms'] = 5



df1_y_kc[0]
df1_x_kc = df1_x_kc.values

label_encoder_zipcode = LabelEncoder()
label_encoder_Renovated = LabelEncoder()
label_encoder_Basement = LabelEncoder()
label_encoder_mes_ano = LabelEncoder()
label_encoder_sazonal = LabelEncoder()
label_encoder_waterfront = LabelEncoder()

df1_x_kc[:,9] = label_encoder_zipcode.fit_transform(df1_x_kc[:,9])
df1_x_kc[:,10] = label_encoder_Basement.fit_transform(df1_x_kc[:,10])
df1_x_kc[:,11] = label_encoder_waterfront.fit_transform(df1_x_kc[:,11])
df1_x_kc[:,12] = label_encoder_Renovated.fit_transform(df1_x_kc[:,12])
df1_x_kc[:,13] = label_encoder_mes_ano.fit_transform(df1_x_kc[:,13])
df1_x_kc[:,14] = label_encoder_sazonal.fit_transform(df1_x_kc[:,14])

OneHotEncoder_hr = ColumnTransformer(transformers=[('Onehot', OneHotEncoder(), [9,10,11,12,13,14])], remainder = 'passthrough')
df1_x_kc = OneHotEncoder_hr.fit_transform(df1_x_kc)
df1_x_kc = df1_x_kc.toarray()

scaler_x_kc = StandardScaler()
df1_x_kc = scaler_x_kc.fit_transform(df1_x_kc)
scaler_y_kc = StandardScaler()
df1_y_kc = scaler_y_kc.fit_transform(df1_y_kc.reshape(-1,1))

novo_registro = df1_x_kc[0].reshape(1,-1)
test =  regressor_rna_kc.predict(novo_registro)
test = test.reshape(1,-1)
df1_y_kc = scaler_y_kc.inverse_transform(df1_y_kc)
test = scaler_y_kc.inverse_transform(test)
print(y_kc_scaled[0])
print(test)
