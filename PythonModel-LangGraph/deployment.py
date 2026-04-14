import time
from mlflow.deployments.databricks import DatabricksDeploymentClient



def create_or_update_endpoint_with_aigateway(
    deployment_client: DatabricksDeploymentClient,
    endpoint_name: str,
    endpoint_config: dict,
    ai_gateway_config: dict,
):
    """
    Create or update a Databricks endpoint with updated AI Gateway.

    This function checks if an endpoint with the given name exists. If it does not exist, it creates a new endpoint
    with the specified configuration. If it does exist, it updates the existing endpoint with the new configuration.

    Args:
        deployment_client (DatabricksDeploymentClient): The Databricks deployment client used to interact with the Databricks API.
        endpoint_name (str): The name of the endpoint to create or update.
        endpoint_config (dict): The configuration for the endpoint.
        ai_gateway_config (dict): The configuration for the AI Gateway.

    Raises:
        Exception: If the endpoint creation or update fails.
    """
    full_config = {
        "ai_gateway": ai_gateway_config,
        "config": endpoint_config,
    }

    endpoint_names = [
        endpoint["name"] for endpoint in deployment_client.list_endpoints()
    ]

    if endpoint_name not in endpoint_names:
        print(f"Creating a new endpoint with name: {endpoint_name}...")
        endpoint = deployment_client.create_endpoint(
            name=endpoint_name,
            config=full_config,
        )
        while True:
            endpoint = deployment_client.get_endpoint(endpoint_name)
            print(endpoint["state"])
            if endpoint["state"]["config_update"] == "UPDATE_FAILED":
                print(endpoint["state"])
                raise Exception(f"Endpoint creation failed, check the logs for more details")
            if endpoint["state"]["ready"] == "READY":
                print(endpoint["state"])
                print(f"Successfully deployed endpoint of name: {endpoint_name}")
                break
            time.sleep(10)
    else:
        print(
            f"Endpoint of name: {endpoint_name} already exists. Attempting to update the endpoint"
        )
        endpoint = deployment_client.update_endpoint_config(
            endpoint=endpoint_name, config=endpoint_config
        )
        endpoint = deployment_client.update_endpoint_ai_gateway(
            endpoint=endpoint_name, config=ai_gateway_config
        )

        while True:
            endpoint = deployment_client.get_endpoint(endpoint_name)
            print(endpoint["state"])
            
            if endpoint["state"]["config_update"] == "UPDATE_FAILED":
                print(endpoint["state"])
                raise Exception(f"Endpoint creation failed, check the logs for more details")

            if (
                endpoint["state"]["config_update"] == "NOT_UPDATING"
                and endpoint["state"]["ready"] == "READY"
            ):
                print(endpoint["state"])
                print(f"Successfully updated endpoint of name: {endpoint_name}")
                break
            time.sleep(10)