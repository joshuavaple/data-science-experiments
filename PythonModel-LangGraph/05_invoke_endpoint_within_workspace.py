# Databricks notebook source
# MAGIC %md
# MAGIC - This notebook demonstrate a method to invoke a workspace endpoint using the token from the notebook context in place of the usual token generated via service principal.
# MAGIC - This method only works internally when you are within the Databricks workspace hosting the endpoint, and is used for endpoint invocation demonstration purpose only.
# MAGIC - Works with serverless compute.

# COMMAND ----------

import requests


def generate_internal_token():
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    return token


def score_model(endpoint_url:str, data_json: str, access_token):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    response = requests.request(method="POST", headers=headers, url=endpoint_url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    return response.json()

# COMMAND ----------

payload = """
{
  "inputs": [
    "On 2026-03-15 at approximately 09:45, a Honda Civic (plate number DEF789) was involved in a collision with a motorcycle (plate number GHI012) on Main Street. The accident resulted in moderate injuries to the motorcyclist. Emergency services arrived quickly, and all individuals were transported to the hospital. No fatalities occurred."
  ]
}
"""

TOKEN = generate_internal_token()
endpoint_url = 'https://adb-3015228593989975.15.azuredatabricks.net/serving-endpoints/gdata_aiworkflowdemo_accident_analysis/invocations'

score_model(endpoint_url=endpoint_url, data_json=payload, access_token=TOKEN)
