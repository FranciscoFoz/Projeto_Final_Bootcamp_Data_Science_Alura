# -*- coding: utf-8 -*-
"""
Notebook_de_Funcoes.ipynb


Original file is located at
    https://colab.research.google.com/drive/1zQMGbtZ3qG_CCBCOZ1Pl6-ea9irlXmW1

#Bootcamp Data Science Alura - Projeto Final

## Requirements
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate,RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.datasets import load_digits
from numpy.ma import MaskedArray
import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray
from skopt import BayesSearchCV

"""
Notebook de funções

1. Funções de manipulação de dados

1.1 remover_pacientes_com_primeira_janela_positiva_para_UTI

"""

def remover_pacientes_com_primeira_janela_positiva_para_UTI(dados):

  '''
  Esta função irá remover os pacientes que possuam resultado positivo(1) 
  para entrada na UTI dentro da primeira janela (0-2)  
  
  dados: dados que o modelo terá como entrada para realizar as transformações nele

  '''
  a_remover = dados.query("WINDOW == '0-2' and ICU == 1")['PATIENT_VISIT_IDENTIFIER'].values            #Query dos pacientes que possuam a primeira janela e resultado positivo
  dados_limpos = dados.query("PATIENT_VISIT_IDENTIFIER not in @a_remover")                              #Query removendo os pacientes na condição acima
  return dados_limpos

"""
1.2 preenche_tabela
"""

def preenche_tabela(dados):

  '''
  Esta função irá preencher a tabela onde esteja NAN com os 
  valores anteriores e depois com os posteriores.
  
  dados: dados que o modelo terá como entrada para realizar as transformações nele
  '''
  features_continuas_colunas = dados.iloc[:,4:-2].columns                                                                                                   #Selecionar colunas de variáveis contínuas
  features_continuas = dados.groupby('PATIENT_VISIT_IDENTIFIER', as_index=False)[features_continuas_colunas].fillna(method='bfill').fillna(method='ffill')  #Preencher os dados NAN da tabela
  features_categoricas = dados.iloc[:,:4]                                                                                                                   #Selecionar as categóricas
  saida = dados.iloc[:, -2:]                                                                                                                                #Selecionar os dados de saída
  dados_finais = pd.concat([features_categoricas,features_continuas,saida],ignore_index=True,axis=1)                                                        #Concatenar os dados anteriores
  dados_finais.columns = dados.columns                                                                                                                      #Renomear as colunas
  return dados_finais

""" 
1.2 prepara_janela
"""

def prepara_janela(linhas):

  '''
  Esta função irá localizar todas as linhas que possuam janela 
  entre "0-2" e UTI igual a 1, retornando as linhas.  

  linhas: linhas que serão transformadas na tabela
                                                 
  '''
  if(np.any(linhas['ICU'])):                                                      
    linhas.loc[linhas['WINDOW']=='0-2','ICU'] = 1                               #Condição se para localizar as linhas com os filtros desejados.
  return linhas.loc[linhas['WINDOW'] == '0-2']


"""
1.3 excluir_coluna_id_paciente
"""

def excluir_coluna_id_paciente(dados):

  '''
  Esta função irá excluir a coluna do ID do paciente.
  
  dados: dados que o modelo terá como entrada para realizar as 
  transformações nele
  '''
  dados_sem_coluna_id_paciente = dados.drop(['PATIENT_VISIT_IDENTIFIER'],axis=1)      #Excluir coluna do id do paciente
  return dados_sem_coluna_id_paciente

"""
1.4 transformar_AGE_PERCENTIL_em_dados_categoricos
"""

def transformar_AGE_PERCENTIL_em_dados_categoricos(dados):
  '''
  
  Esta função irá transformar a coluna AGE Percentil em dados categóricos.
  
  dados: dados que o modelo terá como entrada para realizar as transformações nele
  '''
  dados.AGE_PERCENTIL = dados.AGE_PERCENTIL.astype('category').cat.codes                              #Transformar coluna AGE PECENTIL em dados categóricos
  return dados

"""
1.5 remover_variaveis_correlacionadas

"""

def remover_variaveis_correlacionadas(dados,valor_de_corte):

  '''
  Esta função irá filtrar os dados entre as variáveis, excluindo: 
  "AGE_ABOVE65","AGE_PERCENTIL", "GENDER", "WINDOW" e "ICU".
  Criará uma matriz de correlação entre eles, permanecendo apenas com 
  o triângulo superior.
  Excluirá as colunas com correção passada como parâmetro.
  
  
  dados: dados que o modelo terá como entrada para realizar as transformações nele
  
  valor_de_corte: valor de corte do índice de correlação das variáveis.
  '''
  matriz_corr = dados.iloc[:,3:-2].corr().abs()                                                               #Filtrar dados correlacionando-os e transformando-os em números absolutos.
  matriz_superior = matriz_corr.where(np.triu(np.ones(matriz_corr.shape),k=1).astype(np.bool))                #Criar matriz superior da matriz de correlação
  excluir = [coluna for coluna in  matriz_superior if any(matriz_superior[coluna] > valor_de_corte)]          #Selecionar colunas dentro da matriz superior que contenham alta correlação

  return dados.drop(excluir,axis=1)                                                                           #Retornar o dataframe sem as colunas com alta correlação

"""
1.6 limpar_colunas_com_valores_unicos
"""

def limpar_colunas_com_valores_unicos(dados):
  
  '''
  Esta função irá excluir as colunas que possuíam valores únicos de -1.
  
  dados: dados que o modelo terá como entrada para realizar as transformações nele

  
  '''
  dados = dados.drop(['ALBUMIN_DIFF','BE_ARTERIAL_DIFF','BE_VENOUS_DIFF','BIC_ARTERIAL_DIFF','BIC_VENOUS_DIFF','BILLIRUBIN_DIFF','BLAST_DIFF','CALCIUM_DIFF',
                            'CREATININ_DIFF','FFA_DIFF','GGT_DIFF','GLUCOSE_DIFF','HEMATOCRITE_DIFF', 'HEMOGLOBIN_DIFF','INR_DIFF','LACTATE_DIFF','LEUKOCYTES_DIFF',
                            'LINFOCITOS_DIFF', 'NEUTROPHILES_DIFF','P02_ARTERIAL_DIFF','P02_VENOUS_DIFF','PC02_ARTERIAL_DIFF','PC02_VENOUS_DIFF','PCR_DIFF','PH_ARTERIAL_DIFF',
                            'PH_VENOUS_DIFF','PLATELETS_DIFF','POTASSIUM_DIFF','SAT02_ARTERIAL_DIFF','SAT02_VENOUS_DIFF','SODIUM_DIFF','TGO_DIFF','TGP_DIFF','TTPA_DIFF',
                            'UREA_DIFF','DIMER_DIFF','BLOODPRESSURE_DIASTOLIC_DIFF', 'BLOODPRESSURE_SISTOLIC_DIFF','HEART_RATE_DIFF', 'RESPIRATORY_RATE_DIFF', 'TEMPERATURE_DIFF','OXYGEN_SATURATION_DIFF',
                            'RESPIRATORY_RATE_MIN','BLOODPRESSURE_DIASTOLIC_MAX'],
                           axis=1)
  return dados

"""
2. Funções de criação de visualizações

2.1 grafico_total_pacientes
"""

def grafico_total_pacientes():
  
  '''
  Esta função criará um gráfico de cartão com o total de pacientes.
  '''

  plt.figure(figsize=(7, 3),                                                    #Ajustando o tamanho
            facecolor='white')                                                  #Escolhendo a cor do fundo

  for spine in plt.gca().spines.values():                                       #Removendo os eixos do gráfico
      spine.set_visible(False)

  plt.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False) #Excluindo os valores dos eixos


  plt.text(0, 1.03,                                                             #Ajustando posição dos eixos
          'Número total de pacientes',                                          #Colocando o título
          fontsize=25,                                                          #Formatando o tamanho  
          color='black',                                                        #Formatando a cor
          weight="bold")                                                        #Formatando em negrito

  plt.text(0, 0.25,                                                             #Ajustando posição dos eixos
          '$\\bf{353}$',                                                        #Colocando número com formatação em negrito com auxílio de regex                     
          fontsize=130,                                                         #Formatando o tamanho
          color='#4682B4')                                                      #Formatando a cor
  plt.show()

"""
2.2 grafico_quantidade_entrada_e_nao_entrada_UTI
"""

def grafico_quantidade_entrada_e_nao_entrada_UTI(dados):

  '''
  Esta função criará um gráfico de com a quantidade de pacientes que 
  deram e não deram entrada na UTI.

  dados: dados que serão transformados dentro da função.
  '''
                                           
  tabela_freq_entrada_UTI = pd.DataFrame(dados['ICU'].value_counts())     #Criar DataFrame com a quantidade
  tabela_freq_entrada_UTI.columns = ['UTI']                                                  #Renomear coluna
  tabela_freq_entrada_UTI.index = ['Não','Sim']                                              #Renomear índice

  tabela_freq_entrada_UTI.T.plot(kind='barh',                                    #Plotar o gráfico
                                      stacked=True,                              #Empilhar as barras
                                      figsize=(15,6),                            #Ajustar o tamanho
                                      color=['#808080','#800000'],               #Escolher a cor
                                      fontsize=15,                               #Ajustar o tamanho da fonte
                                      legend=None)                               #Excluir legenda

  for spine in plt.gca().spines.values():                                        #Remover os eixos do gráfico
      spine.set_visible(False)

  quantidade_sem_entrada = tabela_freq_entrada_UTI['UTI'][0]                                 
  quantidade_com_entrada = tabela_freq_entrada_UTI['UTI'][1]

  sem_entrada_percentual = str(round(quantidade_sem_entrada/tabela_freq_entrada_UTI['UTI'].sum(),2)*100)   
  com_entrada_percentual = str(round(quantidade_com_entrada/tabela_freq_entrada_UTI['UTI'].sum(),2)*100)    


                        
  plt.text(quantidade_sem_entrada/2.2,0,                                         #Escolher a posição x e y do texto de %
          sem_entrada_percentual + '%',                                          #Plotar o texto com o %
          fontweight ='bold',                                                    #Formatar para negrito
          fontsize = 20)                                                         #Aumentar a fonte

  plt.text(tabela_freq_entrada_UTI['UTI'].sum() - quantidade_sem_entrada/1.8,0,  #Escolher a posição x e y do texto
          com_entrada_percentual + '%',                                          #Plotar o texto com o %
          fontweight ='bold',                                                    #Formatar para negrito
          fontsize = 20)                                                         #Aumentar a fonte

  plt.yticks([])                                                                 #Excluir valores do eixo y
  plt.xticks([])                                                                 #Excluir valores do eixo x

  plt.text(-0.3, -0.4,                                                           #Escolher posição   
          'Do total de $\\bf{ 353 \\ pacientes}$.' ,                             #Texto com regex
          fontsize=25)                                                           #Escolher tamanho do texto                                            

  plt.text(-0.3, -0.5,                                                           #Escolher posição  
          '$\\bf{ 190 \\ não \\ tiveram \\ entrada}$.' ,                         #Texto com regex
          fontsize=25,                                                           #Escolher tamanho do texto 
          color='#808080')                                                       #Escolher cor

  plt.text(-0.3, -0.6,                                                           #Escolher posição
          'e $\\bf{ 163 \\ tiveram \\ entrada}$.' ,                              #Texto com regex
          fontsize=25,                                                           #Escolher tamanho do texto
          color='#800000')                                                       #Escolher cor
          
  plt.title('Quantidade de pacientes com entrada na UTI',  #Colocar o título
            fontsize=30,                                                         #Tamanho do texto
            loc='left',                                                          #Ajustar do lado esquerdo
            color='black')                                                       #Escolher a cor

  plt.show()
    
"""
2.3 grafico_abaixo_acima_65anos
"""

def grafico_abaixo_acima_65anos(dados):

  '''
  Esta função criará o gráfico comparativo entre os pacientes abaixo e 
  acima de 65 anos de idade que tiveram ou não entrada na UTI.
  
  dados: dados que serão transformados dentro da função.
  '''

  tabela_freq = dados[['AGE_ABOVE65','ICU']].value_counts().reset_index()  #Criar DataFrame com a quantidade
  tabela_freq['AGE_ABOVE65'] = tabela_freq['AGE_ABOVE65'].map({0:'Abaixo',1:'Acima'})         #Mapear para substituir os valores de 0 e 1
  tabela_freq.columns = ['ACIMA_65_ANOS','UTI','QTD']                                         #Modificar as colunas
  tabela_freq = pd.pivot_table(tabela_freq,'QTD','ACIMA_65_ANOS','UTI')                       #Pivotar a tabela


  g = tabela_freq.plot(kind='bar',                                                         #Criar gráfico
                      color=['#808080']+['#800000'],                                       #Escolher cores das barras
                      figsize=(15, 8),                                                     #Escolher tamanho da figura
                      ylabel=False,                                                        #Excluir rótulo do eixo y y
                      legend=None)                                                         #Excluir legenda

  for rotulo in g.containers:                                                              #Criar rótulo de dados nas barras
      g.bar_label(rotulo,fontsize=15,padding=5)

  for spine in plt.gca().spines.values():                                                  #Remover os eixos do gráfico
      spine.set_visible(False)
  

  
  plt.legend(['Não entrada','Entrada'])                                                    #Inserir legenda no gráfico
  plt.xticks(rotation=0, fontsize=15)                                                      #Excluir valores do eixo x
  plt.yticks([])                                                                           #Aumentar a fonte do eixo y
  plt.xlabel(None)                                                                         #Excluir rótulo do eixo y
  plt.ylabel(None)                                                                         #Excluir rótulo do eixo y



  plt.title('Comparativo entre pacientes abaixo e acima de 65 anos'+ 2*('\n'),fontsize=25,loc='left')           #Colocar o título
  plt.text(-0.5,145,'A entrada na UTI é maior em pacientes com idade superior',fontsize=20,color='#800000')     #Colocar o subtítulo

  plt.show()

"""
2.4 faixa_etaria_entrada_UTI
"""

def faixa_etaria_entrada_UTI(dados):

  '''
  Esta função criará o gráfico comparativo entre as faixas etárias 
  entre os pacientes que tiveram ou não entrada na UTI.

  dados: dados que o modelo terá como entrada para realizar as
  transformações nele
  '''

  rotulo = {5:'51-60 anos', 8:'81-90 anos', 0:'0-10 anos', 3:'31-40 anos', 6:'61-70 anos', 1:'11-20 anos', 4:'41-50 anos', 7:'71-80 anos',2:'21-30 anos', 9:'Acima de 90 anos'}  #Rótulos da faixa etária
  tabela_freq = pd.DataFrame(data={'FAIXA_ETARIA': list(dados['AGE_PERCENTIL']), 'UTI': list(dados['ICU'])})                                     #Criar dataframe
  tabela_freq['FAIXA_ETARIA'] = tabela_freq['FAIXA_ETARIA'].map(rotulo)                                                                                                          #Mapear valores dos rótulos
  tabela_freq = tabela_freq.value_counts().reset_index()                                                                                                                         #Criar frequência dos valores
  tabela_freq.columns = ['FAIXA_ETARIA','UTI','QTD']                                                                                                                             #Renomear colunas
  tabela_freq = pd.pivot_table(tabela_freq,'QTD','FAIXA_ETARIA','UTI')                                                                                                           #Pivotar a tabela

  g = tabela_freq.plot(kind='bar',                                              #Criar gráfico
                        color=['#808080']+['#800000'],                          #Escolher cores das barras
                        figsize=(25, 5),                                        #Escolher tamanho da figura
                        ylabel=False,                                           #Excluir rótulo do eixo y y
                        legend=None)                                            #Excluir legenda

  for rotulo in g.containers:                                                   #Criar rótulo de dados nas barras
        g.bar_label(rotulo,fontsize=15,padding=5)


  for spine in plt.gca().spines.values():                                       #Remover os eixos do gráfico
        spine.set_visible(False)
  

  
    
  plt.legend(['Não entrada','Entrada'],                                         #Inserir legenda no gráfico
             bbox_to_anchor=(0.99, 1.1),title='UTI')                            #Localização da legenda e título
  plt.xticks(rotation=0, fontsize=15)                                           #Excluir valores do eixo x
  plt.yticks([])                                                                #Aumentar a fonte do eixo y
  plt.xlabel(None)                                                              #Excluir rótulo do eixo y
  plt.ylabel(None)                                                              #Excluir rótulo do eixo y



  plt.title('Quantidade de pacientes que entraram ou não na UTI por faixa etária'+ 2*('\n'),fontsize=25,loc='left')           #Colocar o título
  plt.text(-0.5, 36,'A partir dos 70 anos a entrada na UTI é superior  ',fontsize=20,color='#800000')                         #Colocar o subtítulo


  plt.show()

"""
2.5 grafico_genero_UTI
"""

def grafico_genero_UTI(dados):

  '''
  Esta função criará o gráfico comparativo entre os 
  generos dos pacientes que tiveram ou não entrada na UTI.

  
  dados: dados que o modelo terá como entrada para realizar as
  transformações nele
  '''

  tabela_freq = pd.DataFrame(data={'FAIXA_ETARIA': list(dados['GENDER']), 'UTI': list(dados['ICU'])})    #Criar dataframe
  tabela_freq['FAIXA_ETARIA'] = tabela_freq['FAIXA_ETARIA'].map({0:'Masculino',1:'Feminino'})            #Mapear valores dos rótulos
  tabela_freq = tabela_freq.value_counts().reset_index()                                                 #Criar frequência dos valores
  tabela_freq.columns = ['GENERO','UTI','QTD']                                                           #Renomear colunas
  tabela_freq = pd.pivot_table(tabela_freq,'QTD','GENERO','UTI')                                         #Pivotar a tabela


  g = tabela_freq.plot(kind='bar',                                               #Criar gráfico
                          color=['#808080']+['#800000'],                         #Escolher cores das barras
                          figsize=(18, 8),                                       #Escolher tamanho da figura
                          ylabel=False,                                          #Excluir rótulo do eixo y y
                          legend=True)                                           #Excluir legenda

  for rotulo in g.containers:                                                    #Criar rótulo de dados nas barras
          g.bar_label(rotulo,fontsize=15,padding=5)


  for spine in plt.gca().spines.values():                                        #Remover os eixos do gráfico
          spine.set_visible(False)
  

  plt.legend(['Não entrada','Entrada'],title='UTI')                              #Localização da legenda e título
  plt.xticks(rotation=0, fontsize=15)                                            #Excluir valores do eixo x
  plt.yticks([])                                                                 #Aumentar a fonte do eixo y
  plt.xlabel(None)                                                               #Excluir rótulo do eixo y
  plt.ylabel(None)                                                               #Excluir rótulo do eixo y


  plt.title('Quantidade de pacientes que entraram ou não na UTI por gênero'+ 2*('\n'),fontsize=25,loc='left')           #Colocar o título
  plt.text(-0.5, 125,'O gênero masculino tem uma maior frequência de entrada na UTI',fontsize=20,color='#800000')       #Colocar o subtítulo


  plt.show()


"""
2.6 grafico_percentual_diferenca_sanguinea_5_menores
"""

def grafico_percentual_diferenca_sanguinea_5_menores(tabela_dados):

  '''
  Esta função criará o gráfico comparativo entre as menores diferenças 
  dos valores sanguíneos entre os pacientes que tiveram entrada na UTI 
  sob o que não tiveram.

  tabela_dados: tabela das diferencas dos dados sanguineos dos pacientes.
  '''
  g = tabela_dados.tail(5).plot(kind='barh',                                      #Escolher cores das barras
                                        color=['#808080'],
                                        figsize=(15, 8),                          #Escolher tamanho da figura
                                        ylabel=False,                             #Excluir rótulo do eixo y y
                                        legend=None)                              #Excluir legenda

  for rotulo in g.containers:                                                     #Criar rótulo de dados nas barras
        g.bar_label(rotulo,fontsize=15,padding=5)

  for spine in plt.gca().spines.values():                                         #Remover os eixos do gráfico
        spine.set_visible(False)
                          
  plt.xticks([])                                                                  #Excluir valores do eixo x
  plt.yticks(fontsize=15)                                                         #Aumentar a fonte do eixo y
  plt.xlabel(None)                                                                #Excluir rótulo do eixo y
  plt.ylabel(None)                                                                #Excluir rótulo do eixo y
  plt.xlim(-70,0)



  plt.text(-80,4.5,'Percentual de diferença dos indicadores sanguíneos dos pacientes'+ 2*('\n'),fontsize=25)               #Colocar o título
  plt.text(-80,4.2,'Top 5 menores entre os pacientes que tiveram entrada pelos que não tiveram  '+ 2*('\n'),fontsize=20)   #Colocar o título

  plt.show()
    
    
"""
2.7 grafico_percentual_diferenca_indicadores_vitais
"""

def grafico_percentual_diferenca_indicadores_vitais(tabela_dados):
  
  '''
  Esta função criará o gráfico comparativo entre as diferenças dos 
  indicadores vitais entre os pacientes que tiveram entrada na UTI 
  sob o que não tiveram.
  
  tabela_dados: tabela das diferencas dos dados vitais dos pacientes.

  '''
  g = tabela_dados.plot(kind='barh',                             
                        color=['#808080'],                      #Escolher cores das barras
                        figsize=(14, 6),                        #Escolher tamanho da figura
                        ylabel=False,                           #Excluir rótulo do eixo y
                        legend=None)                            #Excluir legenda

  for rotulo in g.containers:                                   #Criar rótulo de dados nas barras
          g.bar_label(rotulo,fontsize=15,padding=5)

  for spine in plt.gca().spines.values():                       #Remover os eixos do gráfico
          spine.set_visible(False)
                            
  plt.xticks([])                                                #Excluir valores do eixo x
  plt.yticks(fontsize=15)                                       #Aumentar a fonte do eixo y
  plt.xlabel(None)                                              #Excluir rótulo do eixo y
  plt.ylabel(None)                                              #Excluir rótulo do eixo y
  plt.xlim(-50,50)



  plt.text(-80,6,'Percentual de diferença dos indicadores vitais dos pacientes'+('\n'),fontsize=25)           #Colocar o título
  plt.text(-80,5.5,'Valores entre os que tiveram entrada para os que não tiveram'+('\n'),fontsize=20)         #Colocar o título

  plt.show()

    
"""
2.8 grafico_percentual_diferenca_sanguinea_5_maiores
"""
def grafico_percentual_diferenca_sanguinea_5_maiores(tabela_dados):
  
  '''
  Esta função criará o gráfico comparativo entre as maiores diferenças 
  dos valores sanguíneos entre os pacientes que tiveram entrada na UTI 
  sob o que não tiveram.

  tabela_dados: tabela das diferencas dos dados sanguineos dos pacientes.

  '''
  tabela_dados.sort_values(by='mean',ascending=True,inplace=True)
  g = tabela_dados.tail(5).plot(kind='barh',                                      #Escolher cores das barras
                                        color=['#808080'],
                                        figsize=(15, 8),                          #Escolher tamanho da figura
                                        ylabel=False,                             #Excluir rótulo do eixo y y
                                        legend=None)                              #Excluir leg


  for rotulo in g.containers:                                                     #Criar rótulo de dados nas barras
        g.bar_label(rotulo,fontsize=15,padding=5)

  for spine in plt.gca().spines.values():                                         #Remover os eixos do gráfico
        spine.set_visible(False)
                          
  plt.xticks([])                                                                  #Excluir valores do eixo x
  plt.yticks(fontsize=15)                                                         #Aumentar a fonte do eixo y
  plt.xlabel(None)                                                                #Excluir rótulo do eixo y
  plt.ylabel(None)                                                                #Excluir rótulo do eixo y



  plt.text(-15,4.5,'Percentual de diferença dos indicadores sanguíneos dos pacientes'+ 2*('\n'),fontsize=25)                #Colocar o título
  plt.text(-15,4.2,'Top 5 maiores entre os pacientes que tiveram entrada pelos que não tiveram  '+ 2*('\n'),fontsize=20)    #Colocar o título

  plt.show()



"""
2.9 grafico_matriz_confusao
"""

def grafico_matriz_confusao(modelo,titulo,dados):
  
  '''
  Esta função criará o gráfico da matriz de confusão de acordo 
  com o modelo escolhido.
  
  modelo: modelo que será rodado dentro da função.
  
  titulo: título do gráfico que aparecerá acima dele.

  dados: dados que serão transformados dentro da função.
  '''
  np.random.seed(354354)

  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(['ICU','WINDOW'],axis=1)
  
  x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.15)



  fig, ax = plt.subplots(figsize=(9,9))
  plt.rcParams.update({'font.size': 14})
  y_pred = modelo.predict(x_test)
  disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                                display_labels=['Não entra na UTI','Entra na UTI'],
                                                cmap='YlGnBu',
                                                ax=ax)
  plt.ylabel('Valores reais')
  plt.xlabel('Valores preditos')
  plt.title(titulo +'\n',fontsize=20,loc='left')
  plt.show()

"""
3. Funções de treino e desenvolvimento do modelo

3.1 roda_modelo
"""

def roda_modelo(modelo,dados):

  '''
  Esta função rodará o modelo, imprimirá o AUC médio e 
  o relatório da classificação.
  
  modelo: modelo que será rodado dentro da função.

  dados: dados que serão transformados dentro da função.
  '''
  np.random.seed(354354)

  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(['ICU','WINDOW'],axis=1)
  
  x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.15)

  modelo.fit(x_train,y_train)
  predicao = modelo.predict(x_test)
  prob_predict = modelo.predict_proba(x_test)

  auc = roc_auc_score(y_test,prob_predict[:,1])
  print(f"AUC {auc}")
  print("\nClassification Report")
  print(classification_report(y_test,predicao))

"""
3.2 roda_n_modelos
"""

def roda_n_modelos(modelo,dados,n):

  '''
  Esta função rodará n vezes o modelo e imprimirá o intervalo do auc médio gerado.
  
  modelo: modelo que será rodado dentro da função.

  dados: dados que serão transformados dentro da função.

  n: quantidade de vezes que o modelo irá rodar.
  '''
  np.random.seed(354354)

  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(['ICU','WINDOW'],axis=1)

  auc_lista = []
  for _ in range(n):

    x_train, x_test, y_train, y_test = train_test_split(x, y,stratify=y,test_size=0.15)

    modelo.fit(x_train,y_train)
    prob_predict = modelo.predict_proba(x_test)
    auc = roc_auc_score(y_test, prob_predict[:,1])
    auc_lista.append(auc)
  
  
  auc_medio = np.mean(auc_lista)
  auc_std = np.std(auc_lista)
  print(f'AUC médio {auc_medio}')
  print(f'Intervalo {auc_medio - 2* auc_std} - {auc_medio + 2* auc_std}')

"""
3.3 otimizar_param_bayesiano
"""

def otimizar_param_bayesiano(modelo,params,dados):

  '''
  Esta função irá gerar os melhores hiperparamentros 
  através de um cálculo da estimativa bayesiana.
  
  modelo: modelo que será rodado dentro da função.
  
  paramns: parametros que serão testados no modelo para verificar qual é o melhor.

  dados: dados que serão transformados dentro da função.

  '''
  np.random.seed(354354)
  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(['ICU','WINDOW'],axis=1)
    
  x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.15)

  opt = BayesSearchCV(
      modelo,
      params)

  opt.fit(x_train, y_train)

  return opt.best_params_
