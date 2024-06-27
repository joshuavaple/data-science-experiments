import logging
import azure.functions as func
# import pandas as pd


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    # create dummy data for testing pandas
    data = {"name": ["John", "Anna", "Peter", "Linda"],
            "location": ["New York", "Paris", "Berlin", "London"],}
    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             f"hi! Data: {data}",
             status_code=200
        )
