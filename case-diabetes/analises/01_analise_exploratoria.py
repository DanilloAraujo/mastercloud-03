# Databricks notebook source
import pandas as pd

# COMMAND ----------

spark_df = spark.read.table("hive_metastore.default.diabetes_prediction_dataset")
df = spark_df.toPandas()

# COMMAND ----------

# MAGIC  %md
# MAGIC ### Exploração e agrupamentos

# COMMAND ----------

df.display()

# COMMAND ----------

df['diabetes'].value_counts()

# COMMAND ----------

df.groupby(['diabetes'])['gender'].value_counts(normalize=True)

# COMMAND ----------

df.groupby(['diabetes'])['age'].value_counts(normalize=True)

# COMMAND ----------

df.groupby(['diabetes'])['hypertension'].value_counts(normalize=True)

# COMMAND ----------

df.groupby(['diabetes'])['heart_disease'].value_counts(normalize=True)

# COMMAND ----------

df.groupby(['diabetes'])['smoking_history'].value_counts(normalize=True)

# COMMAND ----------

df.groupby(['diabetes'])['bmi'].value_counts(normalize=True)

# COMMAND ----------

df.groupby(['diabetes'])['HbA1c_level'].value_counts(normalize=True)

# COMMAND ----------

df.groupby(['diabetes'])['blood_glucose_level'].value_counts(normalize=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Correlacao

# COMMAND ----------

import seaborn as sb

# COMMAND ----------

sb.heatmap(df.corr())

# COMMAND ----------

from pandas_profiling import ProfileReport

# COMMAND ----------

profile = ProfileReport(df)

# COMMAND ----------

profile

# COMMAND ----------

df.drop_duplicates()

# COMMAND ----------

spark_df.write.saveAsTable("hive_metastore.default.diabetes_train_gold")
