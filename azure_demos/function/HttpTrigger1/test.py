import logging
import azure.functions as func
from . import utils
from .pkg_1 import utils_1


def sum_numbers(num1: int, num2: int) -> int:
    return num1 + num2

def run_app(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    num1 = 10
    num2 = 20 
    num3 = sum_numbers(num1, num2)
    num4 = utils.multiply_numbers(num1, num2)
    num5 = utils_1.divide_numbers(num1, num2)

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
             f"Hi! The sum of {num1} and {num2} is {num3}.\nThe product of {num1} and {num2} is {num4}.\nThe division of {num1} and {num2} is {num5}.",
             status_code=200
        )
