# type: ignore
import os
from typing import TextIO

import openai
import pandas as pd
import streamlit as st
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.llms import OpenAI

openai.api_key = st.secrets["OPENAI_API_KEY"]


def get_answer_csv(file: TextIO, query: str) -> str:
    """
    Returns the answer to the given query by querying a CSV file.

    Args:
    - file (str): the file path to the CSV file to query.
    - query (str): the question to ask the agent.

    Returns:
    - answer (str): the answer to the query from the CSV file.
    """
    # Load the CSV file as a Pandas dataframe
    # df = pd.read_csv(file)
    #df = pd.read_csv("titanic.csv")

    # Create an agent using OpenAI and the Pandas dataframe
    agent = create_csv_agent(OpenAI(temperature=0), file, verbose=False)
    #agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=False)

    # Run the agent on the given query and return the answer
    #query = "whats the square root of the average age?"
    # query_final = f"""1. {query}. 
    #     2. Respond with scripts without any text. 
    #     3. Respond in plain text code. 
    #     4. Don’t start your response with “Sure, here are”. 
    #     5. Start your response with “import”.
    #     6. Don’t give me any explanation about the script. Response only with python code in a plain text.
    #     7. Use Try and Except for each syntax.
    #     8. Provide minimalist and aesthetic visualization, with flexibility of user to setting the parameter on streamlit.
    #     9. Pay attention to the dataframe schema, don't do any convert."""
    answer = agent.run(query)
    return answer
