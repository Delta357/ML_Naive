# ML_Naive
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
