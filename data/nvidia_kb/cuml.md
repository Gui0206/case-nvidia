---
technology: cuML
category: Machine Learning Acelerado por GPU
aliases: cuML, scikit-learn em GPU, ML clássico acelerado
source: https://docs.rapids.ai/api/cuml/stable/
use_cases: treinar ML clássico em GPU, clustering, regressão, random forest, UMAP, t-SNE
---

# cuML

## O que é
cuML é a biblioteca de **machine learning do RAPIDS**, com API **compatível com scikit-learn**.
Implementa em GPU algoritmos como **Random Forest, regressão linear/logística, KMeans, DBSCAN,
KNN, PCA, UMAP e t-SNE**, com aceleração de treino e inferência. Há também o acelerador
`cuml.accel` (zero-code-change) para scikit-learn, UMAP e HDBSCAN.

## Por que importa para startups AI-native
Nem todo problema é LLM. Para **ML tabular, clustering e redução de dimensionalidade** em
escala, cuML reduz drasticamente o tempo de treino e habilita iteração rápida com o mesmo
código scikit-learn.

## Quando recomendar (sinais na startup)
- Usa **scikit-learn** e treino está lento por volume de dados.
- Casos de **scoring, churn, fraude, recomendação clássica, segmentação**.
- Precisa de **UMAP/t-SNE/clustering** em milhões de pontos.

## Integra com
cuDF, RAPIDS, scikit-learn, CUDA.
