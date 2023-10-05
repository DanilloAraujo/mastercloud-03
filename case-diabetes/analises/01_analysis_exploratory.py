# Databricks notebook source
import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport

# COMMAND ----------

spark_df = spark.read.table("hive_metastore.default.diabetes_prediction_dataset")
df = spark_df.toPandas()

# COMMAND ----------

#Dimensão do conjunto de dados antes da verificação de duplicidade.
df.shape

# COMMAND ----------

#Verificação se existe dado duplicado.
df.duplicated().sum()

# COMMAND ----------

#Remoção dos dados duplicados.
df.drop_duplicates(inplace=True)

# COMMAND ----------

#Dimensão do conjunto de dados após a remoção de duplicidades.
df.shape

# COMMAND ----------

#Verificação de registro nulos.
df.isnull().sum()

# COMMAND ----------

#Informação dos tipos de dados da tabela.
df.info()

# COMMAND ----------

df.describe()

# COMMAND ----------

display(df)

# COMMAND ----------

correlation_matrix = df.corr(numeric_only=True)

# COMMAND ----------

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# COMMAND ----------

profile = ProfileReport(df)

# COMMAND ----------

report_html = profile.to_html()
displayHTML(report_html)

# COMMAND ----------

spark.createDataFrame(df).write.saveAsTable("hive_metastore.default.diabetes_train_gold", mode='overwrite')
#spark_df.write.saveAsTable("hive_metastore.default.diabetes_train_gold", mode='overwrite')
