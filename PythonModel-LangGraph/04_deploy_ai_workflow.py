# Databricks notebook source
# MAGIC %pip install "mlflow[databricks]>=3.1.0" -q
# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import mlflow
from mlflow.deployments import get_deploy_client
from deployment import create_or_update_endpoint_with_aigateway


SECRET_SCOPE = "demo-scope"

mlflow_client = mlflow.tracking.MlflowClient()
deployment_client = get_deploy_client("databricks")

# LLM Endpoint
azure_openai_chat_deployment_name = "gpt-4o"
azure_openai_api_version = "2024-10-21"
model_version = "2024-11-20"
temperature = "0"

# Catalog and Schema
catalog = "workspace"
schema_model = "feature_model"
schema_output = "gold"

# Registry model for deployment
project_name = "aiworkflowdemo"
model_name = "accident_analysis"
project_model_name = f"{project_name}_model_{model_name}"
registered_model_name = f"{catalog}.{schema_model}.{project_model_name}"

# Endpoint Name
endpoint_name = f"gdata_{project_name}_{model_name}"

# Endpoint config
alias_deployed_model = "champion"
workload_size = "Small"
scale_to_zero_enabled = True

# Secret Scope for deployment
secret_scope = "demo-scope"
secret_llm_endpoint = "azure_openai_endpoint"
secret_llm_api_key = "azure_openai_api_key"

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
                "AZURE_OPENAI_ENDPOINT": f"{{{{secrets/{secret_scope}/{secret_llm_endpoint}}}}}",
                "AZURE_OPENAI_API_KEY": f"{{{{secrets/{secret_scope}/{secret_llm_api_key}}}}}",
                "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": azure_openai_chat_deployment_name,
                "AZURE_OPENAI_API_VERSION": azure_openai_api_version,
                "MODEL_VERSION": model_version,
                "TEMPERATURE": temperature,
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
