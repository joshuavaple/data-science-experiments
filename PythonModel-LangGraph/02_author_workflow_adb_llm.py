# Databricks notebook source
# %pip install --upgrade "mlflow[databricks]" "databricks-connect>=16.1" "python-dotenv==1.1.1" "langchain>=1.2" "langchain-openai" "langchain-community" "langgraph>=1.0" "grandalf" "backoff>=2.2.0" "uv" "databricks-agents" "python-dotenv==1.1.1" -q

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]" "langchain>=1.2" "langchain-openai" "langgraph>=1.0" "grandalf" "backoff>=2.2.0" "databricks-langchain" -q
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import os


mlflow.langchain.autolog()

# Change to your ws secret scope:
SECRET_SCOPE = "demo-scope"

# store all to env variables
# Databricks LLM:
os.environ["DATABRICKS_HOST"] = "https://dbc-564fb500-5a75.cloud.databricks.com/"
os.environ["DATABRICKS_MODEL"] = "databricks-gpt-oss-20b"
os.environ["TEMPERATURE"] = "0"
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(scope=SECRET_SCOPE, key="gpt_oss_20b_databricks_token")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Workflow Graph
# MAGIC Added RoadAccidentDetection, and pass-through node `road_accident_absent` that returns the state and that point, connected to END node.

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

# Authentication
from databricks.sdk import WorkspaceClient


workspace_client = WorkspaceClient(
    host=os.environ.get("DATABRICKS_HOST"), token=os.environ.get("DATABRICKS_TOKEN")
)
llm_client = ChatDatabricks(
    model=os.environ.get("DATABRICKS_MODEL"),
    temperature=int(os.environ.get("TEMPERATURE")),
    workspace_client=workspace_client,
)

# ====================================LANGGRAPH CORE============================================
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
location_types = [
    "Expressways/Freeways",
    "Arterials/Main Roads",
    "Collector/Residential Roads",
    "Rural Roads",
    "Others",
]

class RoadAccidentDetection(BaseModel):
    with_road_accident: str = Field(
        description="Whether or not the input contains road traffic accident description. Ignore if there is no accident, or if the accident is not about road traffic.",
        enum=["yes", "no"],
    )
    reason: str = Field(
        description="A short rationale to explain the yes/no answer to the presence of road traffic accident.",
    )

class SingleAccident(BaseModel):
    title: str = Field(
        # default="NO SAFETY INCIDENT",
        description="A short title (below 20 words) to describe the safety accident or accident.",
    )
    date: Optional[str] = Field(
        description="The date the accident occurred if available, reformatted into 'YYYY-MM-DD'.",
    )
    time: Optional[str] = Field(
        description="The time the accident occurred if available, reformatted into 'HH:MM'.",
    )
    reported_injury: str = Field(
        description="Whether or not there is any reported personal injury in the description.",
        enum=["yes", "no"],
    )
    reported_fatality: str = Field(
        description="Whether or not there is any reported fatality in the description.",
        enum=["yes", "no"],
    )

class SingleVehicle(BaseModel):
    """Information of a single vehicle"""
    vehicle_id: Optional[str] = Field(
        description="The ID number or official name of the vehicle inside the input text such as license plate number, crane number...",
    )
    vehicle_type: Optional[str] = Field(
        description="The type of the vehicle inside the input text.",
        enum=vehicle_types,
    )

class MultiVehicle(BaseModel):
    """Information of multiple vehicles"""
    accident_vehicle: List[SingleVehicle]

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
    accident_location: List[SingleLocation]

class InputState(TypedDict):
    input_text: str

class OutputState(TypedDict):
    with_road_accident: str
    reason: str
    accident_overview: Annotated[list[dict], operator.add]
    vehicle_analysis: Annotated[list[dict], operator.add]
    location_analysis: Annotated[list[dict], operator.add]

class OverallState(InputState, OutputState):
    pass


class AccidentAnalysisGraph:
    """
    A class to build and execute a road traffic accident analysis graph to detect and analyze accidents from input texts.
    """
    def __init__(
        self, 
        llm_client: ChatDatabricks, 
        max_concurrency: int = 10,
        max_retries: int = 5,
        timeout: float = 60,
    ):
        self.llm_client = llm_client
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.timeout = timeout

    def detect_road_accident(self, state: InputState) -> OverallState:
        instruction_summary = "you are an expert in road traffic accident analysis and reporting. Your task is to detect if there is any road traffic accident mentioned. Road traffic accidents happen exclusively on road traffic conditions like streets, highway, or bridges, causing property damage, traffic disruption, injuries and fatalities."

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    instruction_summary,
                ),
                ("human", "{input_text}"),
            ]
        )
        chain = prompt_template | self.llm_client.with_structured_output(
            schema=RoadAccidentDetection,
        )
        response = chain.invoke({"input_text": state["input_text"]})
        road_accident_detection = response.model_dump()
        with_road_accident = road_accident_detection["with_road_accident"]
        reason = road_accident_detection["reason"]
        return {"with_road_accident": with_road_accident, "reason": reason}
    
    def check_road_accident(self, state: OverallState) -> OverallState:
        """
        Internal function to perform conditional based on the value of the state field `with_road_accident`
        """
        if state["with_road_accident"] == "yes":
            return "yes"
        elif state["with_road_accident"] == "no":
            return "no"
        else:
            print(
                "WARNING: invalid value for the field with_road_accident, returned 'no'"
            )
            return "no"
    def road_accident_absent(self, state: OverallState):
        # pass-through without updating any field
        return {}

    def extract_accident(self, state: OverallState) -> OverallState:
        instruction_accident = "you are an expert in road traffic accident analysis and reporting. Your task is to extract the accident information from the input text, and parse the accident information into the required structure. Only extract details involving vehicles and people that are directly involved in the accident. Ignore other information mentioned in the writeup."
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    instruction_accident,
                ),
                ("human", "{input_text}"),
            ]
        )
        chain = prompt_template | self.llm_client.with_structured_output(
            schema=SingleAccident
        )
        response = chain.invoke(
            {"input_text": state["input_text"]}
        )
        return {"accident_overview": [response.model_dump()]}
    
    def extract_vehicle(self, state: OverallState) -> OverallState:
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
        return {"vehicle_analysis": [response.model_dump()]}

    def extract_location(self, state: OverallState) -> OverallState:
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
    
    def build_graph(self):
        graph_builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
        graph_builder.add_node("detect_road_accident", self.detect_road_accident)
        graph_builder.add_node("road_accident_absent", self.road_accident_absent)
        graph_builder.add_node("extract_accident", self.extract_accident)
        graph_builder.add_node("extract_vehicle", self.extract_vehicle)
        graph_builder.add_node("extract_location", self.extract_location)
      
        graph_builder.add_edge(START, "detect_road_accident")
        graph_builder.add_conditional_edges(
            source="detect_road_accident",
            path=self.check_road_accident,
            path_map={"yes": "extract_accident", "no": "road_accident_absent"},
        )     
        graph_builder.add_edge("extract_accident", "extract_vehicle")
        graph_builder.add_edge("extract_accident", "extract_location")
        graph_builder.add_edge("extract_vehicle", END)
        graph_builder.add_edge("extract_location", END)
        graph_builder.add_edge("road_accident_absent", END)
        
        graph = graph_builder.compile()
        graph.get_graph().print_ascii()
        self.graph = graph

    def invoke(self, state: OverallState):
        """
        Synchronously invokes the graph.

        Parameters
        -------
            state (OverallState): The initial state for the graph.

        Returns
        -------
            dict: The final state after processing.
        """
        return self.graph.invoke(state)

    async def ainvoke(self, input_text: str):
        """
        Asynchronously invokes the translation graph for a single input text.

        Parameters
        -------
            input_text (str): The input text to process.

        Returns
        -------
            dict: The final state after processing.
        """
        return await asyncio.wait_for(
            self.graph.ainvoke({"input_text": input_text}),
            timeout=self.timeout,
        )

    async def ainvoke_batch(self, input_texts: list[str]):
        """
        Asynchronously invokes the translation graph for a batch of input texts.

        Parameters
        -------
            input_texts (list[str]): List of input texts to process.

        Returns
        -------
            list: List of final states after processing each input.
        """
        tasks = [self.ainvoke(input_text) for input_text in input_texts]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
graph = AccidentAnalysisGraph(llm_client)
graph.build_graph()

# COMMAND ----------

input_text = "On 2026-02-09 at approximately 14:30, a Toyota RAV4 (plate number ABC123) collided with a large truck (plate number XYZ456) on the Express 101. The accident resulted in minor injuries to the driver of the Toyota RAV4. Emergency services responded promptly, and no fatalities were reported."
output = graph.invoke({"input_text": input_text})
output

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. PythonModel Wrapper

# COMMAND ----------

class AccidentAnalyzer(PythonModel):
    """Non-conversational agent for road accident analysis using MLflow model serving."""
    def __init__(self, graph) -> None:
        """Initialize the document analyzer.

        Sets up logging configuration, initializes model properties, and prepares
        the model for serving.
        """
        self.model_name = "road_accident_analyzer"
        self.graph = graph
    def predict(self, model_input: list[str]) -> list[dict]:
        return [self.graph.invoke({"input_text": input_text}) for input_text in model_input]

# COMMAND ----------

model = AccidentAnalyzer(graph)
output = model.predict(["On 2026-02-09 at approximately 14:30, a Toyota RAV4 (plate number ABC123) collided with a large truck (plate number XYZ456) on the Express 101. The accident resulted in minor injuries to the driver of the Toyota RAV4. Emergency services responded promptly, and no fatalities were reported."])
output
