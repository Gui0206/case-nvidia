---
technology: cuDF
category: Data Science Acelerada por GPU
aliases: cuDF, cudf.pandas, DataFrame em GPU
source: https://docs.rapids.ai/api/cudf/stable/
use_cases: acelerar pandas, ETL de DataFrames, processamento tabular em GPU
---

# cuDF

## O que é
cuDF é a biblioteca de **DataFrames acelerada por GPU** do RAPIDS, com API **compatível com
pandas**. Com o modo **`cudf.pandas`**, é possível acelerar código pandas existente
**sem alterar uma linha** (`%load_ext cudf.pandas`), caindo de volta para a CPU quando preciso.

## Por que importa para startups AI-native
É a forma de menor atrito para ganhar performance: a startup mantém o código pandas e ganha
aceleração de GPU em **carregamento, joins, group-bys e transformações** de grandes datasets.

## Quando recomendar (sinais na startup)
- Times de dados que já vivem em **pandas** e batem em limite de performance.
- ETL/feature engineering pesado sobre datasets grandes.
- Notebooks de exploração lentos por causa do volume de dados.

## Integra com
RAPIDS, cuML, Dask, Polars, CUDA.
