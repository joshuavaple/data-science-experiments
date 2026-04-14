# Databricks notebook source
# %pip install --upgrade "mlflow[databricks]" "databricks-connect>=16.1" "python-dotenv==1.1.1" "langchain>=1.2" "langchain-openai" "langchain-community" "langgraph>=1.0" "grandalf" "backoff>=2.2.0" "uv" "databricks-agents" "python-dotenv==1.1.1" -q

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]" "langchain>=1.2" "langchain-openai" "langgraph>=1.0" "grandalf" "backoff>=2.2.0" "uv" -q
# MAGIC %restart_python

# COMMAND ----------

import mlflow
from pkg_resources import get_distribution
import os
from databricks.sdk import WorkspaceClient


os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4o"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-10-21"
os.environ["MODEL_VERSION"] = "2024-11-20"
os.environ["TEMPERATURE"] = "0"

SECRET_SCOPE = "demo-scope"
os.environ["AZURE_OPENAI_ENDPOINT"] = dbutils.secrets.get(scope=SECRET_SCOPE, key="azure_openai_endpoint")
os.environ["AZURE_OPENAI_API_KEY"] = dbutils.secrets.get(scope=SECRET_SCOPE, key="azure_openai_api_key")

catalog = "workspace"
schema = "feature_model"
project_name = "aiworkflowdemo"
model_name = "accident_analysis"
project_model_name = f"{project_name}_model_{model_name}"
registered_model_name = f"{catalog}.{schema}.{project_model_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logging the Workflow as Model from Codes
# MAGIC - The AI workflow will make the prediction using the input example during the process in produce the output example, then the model signature from both.
# MAGIC - You can optionally set an external experiment using `mlflow.set_experiment(EXPERIMENT_NAME)`. Else the notebook experiment will be auto-logged. 

# COMMAND ----------

sample_input = 'On 2026-02-09 at approximately 14:30, a Toyota RAV4 (plate number ABC123) collided with a large truck (plate number XYZ456) on the Express 101. The accident resulted in minor injuries to the driver of the Toyota RAV4. Emergency services responded promptly, and no fatalities were reported.'

with mlflow.start_run() as run:
    model_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model="ai_workflow.py",
        input_example=[sample_input], # model predict() expects a list of strings
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-Deployment Validation
# MAGIC Reload the workflow from the run and make a prediction

# COMMAND ----------

sample_output = mlflow.models.predict(
    model_uri=f"runs:/{model_info.run_id}/{model_name}",
    input_data=[sample_input],
    env_manager="uv", # mlflow recommends using uv for better performance
)
sample_output

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register and Assign Alias to the Workflow

# COMMAND ----------

registered_model_info = mlflow.register_model(
    model_uri=model_info.model_uri, name=registered_model_name
)

# COMMAND ----------

mlflow_client = mlflow.MlflowClient()

mlflow_client.set_registered_model_alias(
    name=registered_model_info.name,
    alias="champion",
    version=registered_model_info.version,
)
