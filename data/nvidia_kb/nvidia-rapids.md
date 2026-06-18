---
technology: NVIDIA RAPIDS
category: Data Science Acelerada por GPU
aliases: RAPIDS, cuGraph, cuSpatial, aceleração de dados
source: https://rapids.ai/
use_cases: pipelines de dados em escala, ETL acelerado, analytics, feature engineering, grafos
---

# NVIDIA RAPIDS

## O que é
RAPIDS é uma **suíte open-source** de bibliotecas que executam **data science e analytics na
GPU**, mantendo APIs familiares do ecossistema PyData. Inclui **cuDF** (DataFrames tipo pandas),
**cuML** (machine learning tipo scikit-learn), **cuGraph** (grafos), **cuSpatial** (geoespacial)
e integrações com Spark, Dask e Polars para escala distribuída.

## Por que importa para startups AI-native
Muitas startups gastam horas em ETL, feature engineering e analytics em CPU. RAPIDS acelera
esses pipelines em **ordens de magnitude**, reduzindo tempo de iteração e custo de
infraestrutura — especialmente em empresas com **grandes volumes de dados tabulares**.

## Quando recomendar (sinais na startup)
- Processa **grandes volumes de dados tabulares** ou de eventos (fintech, adtech, varejo).
- Pipelines de ETL/feature engineering **lentos** ou caros em CPU.
- Treina modelos clássicos de ML em milhões/bilhões de linhas.
- Usa Spark/Dask e quer acelerar com GPU sem reescrever tudo.

## Integra com
cuDF, cuML, cuGraph, Apache Spark, Dask, NVIDIA AI Enterprise, CUDA.
