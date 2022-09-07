# Bootcamp Data Science Alura - Projeto Final
# Previsão da admissão na UTI: <br> Um modelo de Machine Learning a partir de dados dos pacientes do Hospital Sírio Libanês

Neste repositório você encontrará o meu projeto final do Bootcamp Data Science 2021-22 da [Alura](https://www.alura.com.br/).

Devido a pandemia de COVID-19, os hospitais ficaram sobrecarregados faltando recursos de saúde como leitos de UTI, profissionais entre outros. 
Houve a necessidade de prever a quantidade necessária de recursos, principalmente de leitos de UTI. 

O objetivo desse projeto foi criar um modelo de Machine Learing que pudesse conseguir otimizar a necessidade real da permissão de entrada na UTI de acordo com dados individuais dos pacientes.


| :placard: Vitrine.Dev |     |
| -------------  | --- |
| :sparkles: Nome        | **Previsão da admissão na UTI**
| :label: Tecnologias | python
| :rocket: URL         | https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura
| :fire: Desafio     |

<!-- Inserir imagem com a #vitrinedev ao final do link -->
![](https://images.unsplash.com/photo-1504439468489-c8920d796a29?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1171&q=80#vitrinedev)
Fonte: </a> <a href="https://unsplash.com/photos/y5hQCIn1c6o"> @gpiron </a>

## Detalhes do projeto

Elaborado por Francico Foz

<a href="https://img.shields.io/badge/author-gustavolq-blue.svg)](https://www.linkedin.com/in/francisco-tadeu-foz/" target="_blank"><img src="https://img.shields.io/badge/-LinkedIn-%230077B5?style=for-the-badge&logo=linkedin&logoColor=white" target="_blank"></a>  

---

## Projeto 


### Conjunto de Dados

Foram utilizados os dados disponibilizados pelo Hospital Sírio-Libanês no [Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19) .

Os dados foram limpos e dimensionados em cada coluna de com o "Min Max Scaler" para ficar entre -1 e 1.

A quantidade de características (colunas) para cada tipo de informação agrupadas são:

Informações demográficas do paciente (03)
Doenças agrupadas anteriores do paciente (09)
Resultados de sangue (36)
Sinais vitais (06)

Totalizando 54 características, que ainda se desdobram em sua média, mediana, max, min etc. 

A coluna ICU é o resultado da informação se a pessoa foi ou não para a UTI, a partir da janela "Window" que é a escala de tempo medido de cada linha.

Como abaixo:

Janela	
0-2	De 0 a 2 horas da admissão
2-4	De 2 a 4 horas da admissão
4-6	De 4 a 6 horas da admissão
6-12	Das 6 às 12 horas da admissão
Acima de 12	Acima de 12 horas da admissão

### Estrutura do projeto

Este projeto realizará as 4 primeiras etapas do workflow de um projeto de Machine Learning, entregando o resultado para ser realizado posteriormente o Deploy e monitoramento.

![](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/raw/main/Imagens/Workflow_ML.jpg)


Desta forma o projeto foi dividido em 3 notebooks:

* 1º Notebook: [**Aquisição e transformação dos dados**](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/blob/main/Notebooks/Aquisicao_e_transformacao_dos_dados.ipynb)
  *    Limpeza
  *    Análise exploratória

* 2º Notebook: [**Desenvolvimento do modelo**](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/blob/main/Notebooks/Desenvolvimento_do_modelo.ipynb)
  *    Treino
  *    Teste

* 3º Notebook: [**Notebooks de funções**](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/blob/main/Notebooks/Notebook_de_Funcoes.ipynb)
  
  Notebook com todas as funções criadas para o projeto.



### **Considerações finais:**

Devido as características do conjunto de dados iniciais, foi realizada diversas etapas na **limpeza e modelagem dos dados**:

1. Excluiu os pacientes que já entraram nas duas primeiras horas na UTI, devido a incerteza se foi coletado os dados antes ou depois da entrada.
2. Foi preenchido os dados faltantes do conjunto a partir do dado anterior e posterior a ele.
3. Verifiquei se o paciente iria para a UTI ou não até a sua última janela e coloquei a informação na linha da primeira janela (0-2). Exclui as linhas com as demais janelas, para que pudesse prever com os resultados a partir das primeiras horas.
4. Trasnformei a variável "AGE_PERCENTIL" em dados categóricos, afim de se tornar uma variável para o modelo.
5. Exclui as variáveis com alta correlação entre elas, para que o modelo não sofra sobreajuste (overfitting).
6. Mantive o gênero dos pacientes, pois de acordo com as literaturas citadas de [Iaccarino et al (2020)](https://doi.org/10.1371/journal.pone.0237297) e [PECKHAM et al (2020)](https://doi.org/10.1038/s41467-020-19741-6) há uma diferença entre os gêneros nas complicações da doença.
7. Exclui as colunas que possuaim valores únicos (apenas 1, no qual foram as medidas "DIFF" das variáveis) em todas as linhas, para que pudesse ter um conjunto de dados mais limpo possível para o modelo.

Na **análise exploratória** foi possível entender melhor as variáveis e o quadro geral:

1. O total de pacientes do conjunto foi de 353.

![](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/blob/main/Imagens/total_pacientes20220219.png?raw=true)

2. Desses 190 (54%) não tiveram entrada e 163 (46%) tiveram.

![](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/blob/main/Imagens/total_entrada_na_UTI20220219.png?raw=true)

3. Pacientes acima de 65 anos de idade tem uma maior frequência de entrada na UTI.

![](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/blob/main/Imagens/abaixo_acima_65_anos20220219.png?raw=true)

4. Ao visualizar todas as faixas etárias, foi possível confirmar que a partir da faixa de 61-70 anos frequência se iguala e a partir dos 70 anos a entrada na UTI é maior.

![](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/blob/main/Imagens/faixa_etaria20220219.png?raw=true)

5. Também pude confirmar a questão da entrada na UTI através do gênero, no qual os homens tem maior incidência de complicações na doença.

![](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/blob/main/Imagens/genero20220219.png?raw=true)

6. Analisei em conjunto todas as doenças anteriores dos pacientes, para pudesse entender ao todo como cada pessoa com estas características se comportaria na entrada ou não na UTI. A maior frequência está em pacientes que não tiveram nenhuma outra doença mas se concentrou em outras doenças. Talvez se esta variável estivesse mais detalhada em outras classificações, poderíamos ter um modelo mais assertivo do ponto de vista das doenças anteriores.

7. Dentre os indicadores sanguíneos, observei os que tiveram maior diferença (positiva e negativa), entre as pessoas que entraram ou não na UTI foram: 
  * Negativa:
    * Sódio
    * Hematócritos
    * Linfóciotos
    * BIC Venoso
    * PH Venoso
 
![](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/blob/main/Imagens/indicadores_sanguineos_5_menores20220219.png?raw=true)
 
  * Positiva:
    * Lactato
    * PCR
    * SAT02 Venoso
    * Cálcio
    * Potássio
 
![](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/blob/main/Imagens/indicadores_sanguineos_5_maiores20220219.png?raw=true)

8. Já nos sinais vitais os dois indicadores que tiveram maior diferença foram o de temperatura (-35%) e D-Dímero (22%) dos pacientes que não entraram na UTI.

![](https://github.com/FranciscoFoz/Projeto_Final_Bootcamp_Data_Science_Alura/blob/main/Imagens/Percentual_diferenca_indicadores_vitais20220219.png?raw=true)


Durante o **desenvolvimento do modelo**, escolhi algumas etapas para definir o melhor:

1. Baseado no artigo científico de [Subudhi et al(2021)](https://doi.org/10.1038/s41746-021-00456-x), escolhi 3 modelos que tiveram maior índice de F1 para realizar os testes do conjunto de dados:
  * Bagging Classifier
  * Gradient Boosting Classifier
  * Ramdom Florest
2. Realizei os testes com cada um deles, visualizando as métricas de AUC, a tabela de precision, recall e f1-score, a matriz de confusão do teste e o AUC médio do intervalo de AUC coletado, ao rodar ele 50 vezes.
3. A partir desse etapa, escolhi os dois modelos que tiveram melhor desempenho de AUC médio e Reccal no caso de entrada na UTI (pois é uma medida muito importante para que o modelo não errasse em pessoas que deveriam entrar na UTI e o modelo indicasse que ela não precisaria).
4. Otimizei os hiperparâmetros do modelo com uma abordagem Bayesiana, procurando os melhores parâmetros. 
5. Realizei os testes das métricas novamente.
6. Os resultados mostraram o Gradient Boosting otimizado com um melhor resultado de AUC médio (0.794) dentre o intervalo, além dele ter tido um reccal em pessoas que deveriam entrar na UTI de 0.79.


Foi um modelo definido com uma métrica média não muito alta, mas ele poderá ser uma importante ferramenta para os profissionais da saúde poderem já se alertarem aos primeiros sinais dos pacientes ao entrarem no hospital.

Como próximos passos seria interessante que ele estivesse fazendo parte de um sistema integrado as ferramentas dos profissionais da saúde, mostrando a previsão do modelo para que o profissional pudesse tomar a decisão.

### **Referências**

IACCARINO, G. et al. Gender differences in predictors of intensive care units admission among COVID-19 patients: The results of the SARS-RAS study of the Italian Society of Hypertension. **PloS one**, v. 15(10), oct. 2020. DOI https://doi.org/10.1371/journal.pone.0237297 . Disponível em: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237297 . Acesso em: 25 jan. 2022.

PECKHAM, H. et al. Male sex identified by global COVID-19 meta-analysis as a risk factor for death and ITU admission. **Nature Communications**, v. 11(6317), dec. 2020. DOI: https://doi.org/10.1038/s41467-020-19741-6 . Disponível em: https://www.nature.com/articles/s41467-020-19741-6#citeas . Acesso em: 25 jan. 2022.

SUBUDHI, S. et al. Comparing machine learning algorithms for predicting ICU admission and mortality in COVID-19. **NPJ Digit**. Med. 4, v. 87, may 2021. DOI: https://doi.org/10.1038/s41746-021-00456-x . Disponível em: https://www.nature.com/articles/s41746-021-00456-x#citeas . Acesso em: 29 jan. 2022
