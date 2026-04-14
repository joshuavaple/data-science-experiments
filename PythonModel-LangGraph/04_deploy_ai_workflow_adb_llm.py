# Databricks notebook source
# MAGIC %pip install "mlflow[databricks]>=3.1.0" -q
# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import mlflow
from mlflow.deployments import get_deploy_client
from deployment import create_or_update_endpoint_with_aigateway
import os


mlflow_client = mlflow.tracking.MlflowClient()
deployment_client = get_deploy_client("databricks")

# Databricks LLM:
DATABRICKS_HOST = "https://dbc-564fb500-5a75.cloud.databricks.com/"
DATABRICKS_MODEL = "databricks-gpt-oss-20b"
TEMPERATURE = "0"

# Catalog and Schema
catalog = "workspace"
schema_model = "feature_model"
schema_output = "gold"

# Registry model for deployment
project_name = "aiworkflowdemo"
model_name = "accident_analysis_adb_llm"
project_model_name = f"{project_name}_model_{model_name}"
registered_model_name = f"{catalog}.{schema_model}.{project_model_name}"

# Endpoint Name
endpoint_name = f"personal_{project_name}_{model_name}"

# Endpoint config
alias_deployed_model = "champion"
workload_size = "Small"
scale_to_zero_enabled = True

# Secret Scope for deployment
# Using token by the "Generate Access Token" button on the AI Gateway model. 
# SP is created but AI gateway foundational model not able to grant access (moving to UC)
SECRET_SCOPE = "demo-scope"
SECRET_DATABRICKS_MODEL = "gpt_oss_20b_databricks_token"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the model from UC using name and alias

# COMMAND ----------

deployed_model_version = mlflow_client.get_model_version_by_alias(
    registered_model_name, alias_deployed_model
)
print(f"Model alias @champion details:")
print(f"Model name: {deployed_model_version.name}")
print(f"Model version: {deployed_model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Set endpoint config

# COMMAND ----------

endpoint_config = {
    "served_entities": [
        {
            "name": f"{model_name}_entity",
            "entity_name": deployed_model_version.name,
            "entity_version": str(deployed_model_version.version),
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero_enabled,
            "workload_type": "CPU",
            "environment_vars": {
                "DATABRICKS_MODEL": DATABRICKS_MODEL,
                "TEMPERATURE": TEMPERATURE,
                "DATABRICKS_HOST": DATABRICKS_HOST,
                "DATABRICKS_TOKEN": f"{{{{secrets/{SECRET_SCOPE}/{SECRET_DATABRICKS_MODEL}}}}}",
            },
        }
    ],
    "traffic_config": {
        "routes": [
            {
                "served_model_name": f"{model_name}_entity",
                "traffic_percentage": 100,
            }
        ]
    },
}

# inference table for Databricks Free Edition workspace is not enabled.
# it will work for paid workspaces
# HTTPError: 500 Server Error: Inference table is not currently supported for this endpoint type in this workspace. for url: https://dbc-564fb500-5a75.cloud.databricks.com/api/2.0/serving-endpoints/. Response text: {"error_code": "FEATURE_DISABLED", "message": "Inference table is not currently supported for this endpoint type in this workspace."}

ai_gateway_config = {
    # "inference_table_config": {
    #     "catalog_name": catalog,
    #     "enabled": "true",
    #     "schema_name": schema_output,
    # },
    "usage_tracking_config": {"enabled": "true"},
}


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create or Update Endpoint
# MAGIC - Attempt 1: no env variables set in endpoint.
# MAGIC   - Deployment error (expected): `mlflow.exceptions.MlflowException: Failed to run user code from /model/ai_workflow_adb_llm.py. Error: default auth: cannot configure default credentials, please check https://docs.databricks.com/en/dev-tools/auth.html#databricks-client-unified-authentication to configure credentials for your preferred authentication method.`
# MAGIC
# MAGIC - Attepmt 2: set only DATABRICKS_HOST to workspace URL
# MAGIC   - Deployment error: `mlflow.exceptions.MlflowException: Failed to run user code from /model/ai_workflow_adb_llm.py. Error: default auth: cannot configure default credentials, please check https://docs.databricks.com/en/dev-tools/auth.html#databricks-client-unified-authentication to configure credentials for your preferred authentication method. Config: host=https://adb-3015228593989975.15.azuredatabricks.net, account_id=b546b988-302d-426c-8b59-b2009e1bfca1, workspace_id=3015228593989975, discovery_url=https://adb-3015228593989975.15.azuredatabricks.net/oidc/.well-known/oauth-authorization-server, azure_tenant_id=bc1b92b9-5dc9-49be-995b-c97eb515a1d3. Env: DATABRICKS_HOST.`
# MAGIC
# MAGIC - Attempt 3:
# MAGIC   - using access token as DATABRICKS_TOKEN and workspace URL as DATABRICKS_HOST in the endpoint env vars

# COMMAND ----------

create_or_update_endpoint_with_aigateway(
    deployment_client=deployment_client,
    endpoint_name=endpoint_name,
    endpoint_config=endpoint_config,
    ai_gateway_config=ai_gateway_config,
)

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_url = f'{workspace_url}/serving-endpoints/{endpoint_name}/invocations'
print(f"INFO: Endpoint URL: {endpoint_url}")
