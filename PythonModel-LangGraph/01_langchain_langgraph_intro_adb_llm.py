# Databricks notebook source
# MAGIC %pip install --upgrade "mlflow[databricks]" "langchain>=1.2" "langchain-openai" "langgraph>=1.0" "databricks-langchain" -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import mlflow
import os


# Change to your ws secret scope:
SECRET_SCOPE = "demo-scope"

# store all to env variables
# Databricks LLM:
os.environ["DATABRICKS_HOST"] = "https://dbc-564fb500-5a75.cloud.databricks.com/"
os.environ["DATABRICKS_MODEL"] = "databricks-gpt-oss-20b"
os.environ["TEMPERATURE"] = "0"
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(scope=SECRET_SCOPE, key="gpt_oss_20b_databricks_token")

# COMMAND ----------

# async
import asyncio

# langgraph:
import operator
from langgraph.types import Send
from langgraph.graph import END, StateGraph, START

# langchain
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Any, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence

# from langchain_openai.chat_models.azure import AzureChatOpenAI
from databricks_langchain import ChatDatabricks

# mlflow
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.tracing import set_destination
from mlflow.entities import SpanType
from mlflow.entities.trace_location import MlflowExperimentLocation

# runtime env
import logging
import os


# autolog langchain to MLflow UI
mlflow.langchain.autolog()

# Init LLM client
# Databricks LLM
from databricks.sdk import WorkspaceClient


workspace_client = WorkspaceClient(
    host=os.environ.get("DATABRICKS_HOST"), token=os.environ.get("DATABRICKS_TOKEN")
)
llm_client = ChatDatabricks(
    model=os.environ.get("DATABRICKS_MODEL"),
    temperature=int(os.environ.get("TEMPERATURE")),
    workspace_client=workspace_client,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. LangChain Basics

# COMMAND ----------

# Simple problem: classify vehicles into pre-defined types
# 1. Define controlled output categories 
vehicle_types = [
    "Passenger Car",
    "Public Bus",
    "Truck",
    "Motorcycle",
    "Bicycle",
    "Emergency Vehicle",
    "Personal Mobility Equipment",
    "Others",
]

# 2. Define Pydantic schema for output structure
class SingleVehicle(BaseModel):
    """Information of a single vehicle"""
    vehicle_id: Optional[str] = Field(
        description="The ID number or official name of the vehicle inside the input text such as license plate number, crane number...",
    )
    vehicle_type: Optional[str] = Field(
        description="The type of the vehicle inside the input text.",
        enum=vehicle_types,
    )

# Use pydantic model to define nested output structure for multiple vehicles
class MultiVehicle(BaseModel):
    """Information of multiple vehicles"""
    vehicles: List[SingleVehicle]

# 3. Define system prompt and prompt template:
# System prompt
instruction = "You are an expert in road traffic accident analysis. extract the vehicle information. Only extract if the information is mentioned in the input text."

# Prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            instruction,
        ),
        ("human", "{input_text}"),
    ]
)

# 4. Define chain. 
# Use `.with_structured_output` to specify the output schema to the LLM
chain = prompt_template | llm_client.with_structured_output(
    schema=MultiVehicle
)

# COMMAND ----------

input_text = "On 2026-02-09 at approximately 14:30, a Toyota RAV4 (plate number ABC123) collided with a large truck (plate number XYZ456) on the Express 101. The accident resulted in minor injuries to the driver of the Toyota RAV4. Emergency services responded promptly, and no fatalities were reported."
output = chain.invoke({"input_text": input_text})
output

# COMMAND ----------

# without the `.with_structured_output`, the chain returns the usual AI message from the chat model
chain_unstructured = prompt_template | llm_client
output = chain_unstructured.invoke({"input_text": input_text})
output

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. LangGraph Basics
# MAGIC - Ref: https://docs.langchain.com/oss/python/langgraph/graph-api
# MAGIC - At its core, LangGraph models agent workflows as graphs. You define the behavior of your agents using three key components:
# MAGIC   - State: A shared data structure that represents the current snapshot of your application. It can be any data type, but is typically defined using a shared state schema.
# MAGIC   - Nodes: Functions that encode the logic of your agents. They receive the current state as input, perform some computation or side-effect, and return an updated state.
# MAGIC   - Edges: Functions that determine which Node to execute next based on the current state. They can be conditional branches or fixed transitions.
# MAGIC - To emphasize: Nodes and Edges are nothing more than functions – they can contain an LLM or just good ol’ code.

# COMMAND ----------

# MAGIC %md
# MAGIC ### i. Nodes

# COMMAND ----------

# Illustrate a graph
from IPython.display import Image, display
from datetime import datetime

# 1. Define a shared state across nodes
# Ref to docs for more info on node-specific state or different states for I/O
# use the reducer function operator.add to add the node output to the state, not replacing it
class OverallState(TypedDict):
    input_text: str
    vehicle_analysis: Annotated[list[dict], operator.add]
    location_analysis: Annotated[list[dict], operator.add]

# 2. Wrap the previous chains to node functions
# if not specified, the output state class is the same as the input state
# explicitly specifying here for clarity

# Node 1
def extract_vehicle(state: OverallState) -> OverallState:
    instruction = "You are an expert in road traffic accident analysis. extract the vehicle information. Only extract if the information is mentioned in the input text."
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                instruction,
            ),
            ("human", "{input_text}"),
        ]
    )
    chain = prompt_template | llm_client.with_structured_output(
        schema=MultiVehicle
    )
    response = chain.invoke({"input_text": state["input_text"]})
    # LangGraph syntax: update the relevant fields of the output state
    return {"vehicle_analysis": [response.model_dump()]}


# Node 2
location_types = [
    "Expressways/Freeways",
    "Arterials/Main Roads",
    "Collector/Residential Roads",
    "Rural Roads",
    "Others",
]

class SingleLocation(BaseModel):
    """Information of a single location"""
    location_name: Optional[str] = Field(
        description="The name of the location explicitly mentioned such as street name, highway name",
    )
    location_type: Optional[str] = Field(
        description="The type of the location explicitly mentioned inside the input text.",
        enum=location_types,
    )

class MultiLocation(BaseModel):
    """Information of multiple locations"""
    locations: List[SingleLocation]


def extract_location(state: OverallState):
    instruction = "You are an expert in road traffic accident analysis. extract the location information. Only extract if the information is mentioned in the input text."
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                instruction,
            ),
            ("human", "{input_text}"),
        ]
    )
    chain = prompt_template | llm_client.with_structured_output(
        schema=MultiLocation
    )
    response = chain.invoke({"input_text": state["input_text"]})
    return {"location_analysis": [response.model_dump()]}

# COMMAND ----------

# MAGIC %md
# MAGIC ### ii. Sequential Nodes in a Graph

# COMMAND ----------

# 3. Define the graph
graph_builder = StateGraph(OverallState)
graph_builder.add_node("extract_vehicle", extract_vehicle)
graph_builder.add_node("extract_location", extract_location)

graph_builder.add_edge(START, "extract_vehicle")
graph_builder.add_edge("extract_vehicle", "extract_location")
graph_builder.add_edge("extract_location", END)

# 4. Compile the graph
seq_graph = graph_builder.compile()
# seq_graph.get_graph().print_ascii()
display(Image(seq_graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

# 6. Invoke the graph
tik = datetime.now()
output = seq_graph.invoke({"input_text": input_text})
tok = datetime.now()
print(f"Time taken: {tok-tik}")
output

# COMMAND ----------

# MAGIC %md
# MAGIC ### iii. Parallel Nodes in a Graph

# COMMAND ----------

# Parallel Nodes if they dont depend on each other
graph_builder = StateGraph(OverallState)
graph_builder.add_node("extract_vehicle", extract_vehicle)
graph_builder.add_node("extract_location", extract_location)

graph_builder.add_edge(START, "extract_vehicle")
graph_builder.add_edge(START, "extract_location") # source = START, instead of "extract_vehicle"
graph_builder.add_edge("extract_vehicle", END)
graph_builder.add_edge("extract_location", END)

par_graph = graph_builder.compile()
display(Image(par_graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

tik = datetime.now()
output = par_graph.invoke({"input_text": input_text})
tok = datetime.now()
print(f"Time taken: {tok-tik}")
output
