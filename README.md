# ML_Naive

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

![Logo](https://img.freepik.com/fotos-gratis/fundo-de-conexoes-de-rede-abstrata_1048-7961.jpg?w=996&t=st=1685900340~exp=1685900940~hmac=82a9718b7a8532866ebf9a101f2629f2cbf7eaa180531f8eb59ff84f42e2e198)


Neste repositório apresentar um estudo sobre o algoritmo Naive Bayes aplicado a um projeto de processamento de linguagem natural. O objetivo deste estudo é analisar a eficácia do algoritmo Naive Bayes na classificação de textos.

## Metodologia
A metodologia utilizada neste estudo é composta por várias etapas. A primeira etapa é o pré-processamento dos dados, que inclui a limpeza dos textos, remoção de caracteres especiais, conversão para letras minúsculas, remoção de stopwords (palavras comuns que não contribuem para a classificação) e lematização (redução de palavras flexionadas ao seu lema). Esse pré-processamento tem como objetivo preparar os dados para o treinamento do modelo.

## Modelo
O modelo Naive Bayes é então aplicado aos dados pré-processados. O algoritmo Naive Bayes é uma técnica de classificação baseada no Teorema de Bayes, que assume independência condicional entre as características (ou palavras) do texto. Esse modelo é treinado usando um conjunto de dados rotulados, onde cada texto está associado a uma categoria pré-definida.

## Métricas
Após o treinamento do modelo, são utilizadas métricas de avaliação para medir sua eficácia. As métricas comumente utilizadas incluem precisão, recall e F1-score. 
A precisão mede a proporção de textos classificados corretamente em relação ao total de textos classificados como positivos. O recall mede a proporção de textos positivos classificados corretamente em relação ao total de textos positivos. O F1-score é uma média harmônica entre precisão e recall.

Além disso, é criada uma matriz de confusão para visualizar o desempenho do modelo. A matriz de confusão mostra o número de textos classificados corretamente e incorretamente para cada categoria.

## Conclusão
Em conclusão, este estudo apresentou a aplicação do algoritmo Naive Bayes em um projeto de processamento de linguagem natural. Através do pré-processamento dos dados, treinamento do modelo e análise das métricas de avaliação, foi possível avaliar a eficácia do algoritmo na classificação de textos. Os resultados obtidos podem ser utilizados como base para futuras melhorias e desenvolvimento de sistemas de classificação mais robustos.

## Stack utilizada

**Programação** Python, R.

**Machine learning**: Scikit-learn.

**Leitura CSV**: Pandas.

**Análise de dados**: Seaborn, Matplotlib.

**Modelo machine learning - Processo de linguagem natural**: NLTK, TextBlob, Vander.

## Dataset

| Dataset               | Link                                                |
| ----------------- | ---------------------------------------------------------------- |
|  |[Projeto - Notebook]()|
|  |[Projeto - Notebook]()|
|  |[Projeto - Notebook]()|
|  |[Projeto - Notebook]()|
|  |[Projeto - Notebook]()|



## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

`API_KEY`

`ANOTHER_API_KEY`


## Instalação

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```

## Demo modelo Naive bayes

```
## Aplicação em python

# Importando biblioteca
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dados de treinamento
texts = ["I love this movie", "This movie is great", "I really enjoyed this movie"]
labels = ["positive", "positive", "positive"]

# Pré-processamento dos textos usando CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Treinamento do modelo Naive Bayes
model = MultinomialNB()
model.fit(X, labels)

# Dados de teste
test_texts = ["This movie is terrible", "I didn't like this movie", "This movie is amazing"]

# Pré-processamento dos textos de teste
X_test = vectorizer.transform(test_texts)

# Classificação dos textos de teste usando o modelo treinado
predicted_labels = model.predict(X_test)

# Resultados
for text, label in zip(test_texts, predicted_labels):
    print(f"Text: {text} --> Label: {label}")
    
## Aplicação em R
# Instalar o pacote 'e1071' se necessário
# install.packages("e1071")

library(e1071)

# Dados de treinamento
texts <- c("Eu amo esse filme", "Esse filme é ótimo", "Eu realmente gostei desse filme")
labels <- c("positivo", "positivo", "positivo")

# Pré-processamento dos textos usando o pacote 'tm'
library(tm)

corpus <- Corpus(VectorSource(texts))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("portuguese"))
corpus <- tm_map(corpus, stripWhitespace)

dtm <- DocumentTermMatrix(corpus)
X <- as.data.frame(as.matrix(dtm))

# Treinamento do modelo Naive Bayes
model <- naiveBayes(X, labels)

# Dados de teste
test_texts <- c("Esse filme é terrível", "Eu não gostei desse filme", "Esse filme é incrível")

# Pré-processamento dos textos de teste
test_corpus <- Corpus(VectorSource(test_texts))
test_corpus <- tm_map(test_corpus, content_transformer(tolower))
test_corpus <- tm_map(test_corpus, removePunctuation)
test_corpus <- tm_map(test_corpus, removeNumbers)
test_corpus <- tm_map(test_corpus, removeWords, stopwords("portuguese"))
test_corpus <- tm_map(test_corpus, stripWhitespace)

test_dtm <- DocumentTermMatrix(test_corpus)
X_test <- as.data.frame(as.matrix(test_dtm))

# Classificação dos textos de teste usando o modelo treinado
predicted_labels <- predict(model, X_test)

# Resultados
for (i in 1:length(test_texts)) {
  cat("Text:", test_texts[i], "--> Label:", predicted_labels[i], "\n")
}

```

## Melhorias

Que melhorias você fez no seu código? 
- Ex: refatorações, melhorias de performance, acessibilidade, etc


## Suporte

Para suporte, mande um email para rafaelhenriquegallo@gmail.com
