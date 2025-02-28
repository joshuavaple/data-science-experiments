# Databricks notebook source
import json
import configparser
import os
import pandas as pd

print('-----CWD-----')
cwd = os.getcwd()
print(cwd)

print('-----relative import-----')
try:
    from utils.demo_utils import MY_CONSTANT, my_empty_function, MyClass
    print(MY_CONSTANT)
    print(my_empty_function())
    my_class = MyClass()
    print(my_class.my_method())
    print(my_class.my_attribute)
except Exception as e:
    print(e)

print('-----load json-----')
try:
    with open('./data/demo_data.json') as f:
        data = json.load(f)
        print(data)
except Exception as e:
    print(e)

print('-----load ini to config-----')
config = configparser.ConfigParser()
config.read('./config/config.ini')
print(config.sections())

print('-----load csv to pandas-----')
try:
    df = pd.read_csv('./data/penguins.csv')
    print(df.head(3))
except Exception as e:
    print(e)

print('-----load csv to spark-----')
try:
    df = spark.read.format("csv").load(f"file:{cwd}/data/penguins.csv")
    display(df.limit(3))
except Exception as e:
    print(str(e)[:100] + '...')
