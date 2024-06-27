from typing import Union

def divide_numbers(a: Union[int, float], b: Union[int, float], rounded_ndigits: int = 2) -> Union[int, float]:
    return round(a / b, rounded_ndigits)