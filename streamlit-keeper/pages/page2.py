import requests
from bs4 import BeautifulSoup
import math
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import asyncio
import nest_asyncio
import json
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
import os
from pinecone import Pinecone, ServerlessSpec
from pprint import pprint
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import os
from groq import Groq
import streamlit as st
from page1 import *
from langchain_community.document_loaders import PyPDFLoader
import getpass
from langchain_community.vectorstores import FAISS
from langchain_community.tools.you import YouSearchTool
from langchain_community.utilities.you import YouSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import chat_agent_executor
from langgraph.checkpoint import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from fpdf import Template
from pdfrw import PdfReader, PdfWriter, PageMerge



st.title("checking")
getInformation = st.session_state["my_input"]
result = getInformation.split(",")
assignment_id, home_state, income = int(result[0]), result[1], float(result[2])
st.write(assignment_id, home_state, income)

taxing = {
    "Alabama": {"tax_code_name": "AL", "tax_form": "Alabama Form 40"},
    "Alaska": {"tax_code_name": "AK", "tax_form": "No state income tax"},
    "Arizona": {"tax_code_name": "AZ", "tax_form": "Arizona Form 140"},
    "Arkansas": {"tax_code_name": "AR", "tax_form": "Arkansas Form AR1000F"},
    "California": {"tax_code_name": "CA", "tax_form": "California Form 540"},
    "Colorado": {"tax_code_name": "CO", "tax_form": "Colorado Form 104"},
    "Connecticut": {"tax_code_name": "CT", "tax_form": "Connecticut Form CT-1040"},
    "Delaware": {"tax_code_name": "DE", "tax_form": "Delaware Form 200-01"},
    "Florida": {"tax_code_name": "FL", "tax_form": "No state income tax"},
    "Georgia": {"tax_code_name": "GA", "tax_form": "Georgia Form 500"},
    "Hawaii": {"tax_code_name": "HI", "tax_form": "Hawaii Form N-11"},
    "Idaho": {"tax_code_name": "ID", "tax_form": "Idaho Form 40"},
    "Illinois": {"tax_code_name": "IL", "tax_form": "Illinois Form IL-1040"},
    "Indiana": {"tax_code_name": "IN", "tax_form": "Indiana Form IT-40"},
    "Iowa": {"tax_code_name": "IA", "tax_form": "Iowa Form IA 1040"},
    "Kansas": {"tax_code_name": "KS", "tax_form": "Kansas Form K-40"},
    "Kentucky": {"tax_code_name": "KY", "tax_form": "Kentucky Form 740"},
    "Louisiana": {"tax_code_name": "LA", "tax_form": "Louisiana Form IT-540"},
    "Maine": {"tax_code_name": "ME", "tax_form": "Maine Form 1040ME"},
    "Maryland": {"tax_code_name": "MD", "tax_form": "Maryland Form 502"},
    "Massachusetts": {"tax_code_name": "MA", "tax_form": "Massachusetts Form 1"},
    "Michigan": {"tax_code_name": "MI", "tax_form": "Michigan Form MI-1040"},
    "Minnesota": {"tax_code_name": "MN", "tax_form": "Minnesota Form M1"},
    "Mississippi": {"tax_code_name": "MS", "tax_form": "Mississippi Form 80-105"},
    "Missouri": {"tax_code_name": "MO", "tax_form": "Missouri Form MO-1040"},
    "Montana": {"tax_code_name": "MT", "tax_form": "Montana Form 2"},
    "Nebraska": {"tax_code_name": "NE", "tax_form": "Nebraska Form 1040N"},
    "Nevada": {"tax_code_name": "NV", "tax_form": "No state income tax"},
    "New Hampshire": {"tax_code_name": "NH", "tax_form": "New Hampshire Form DP-10"},
    "New Jersey": {"tax_code_name": "NJ", "tax_form": "New Jersey Form NJ-1040"},
    "New Mexico": {"tax_code_name": "NM", "tax_form": "New Mexico Form PIT-1"},
    "New York": {"tax_code_name": "NY", "tax_form": "New York Form IT-201"},
    "North Carolina": {"tax_code_name": "NC", "tax_form": "North Carolina Form D-400"},
    "North Dakota": {"tax_code_name": "ND", "tax_form": "North Dakota Form ND-1"},
    "Ohio": {"tax_code_name": "OH", "tax_form": "Ohio Form IT-1040"},
    "Oklahoma": {"tax_code_name": "OK", "tax_form": "Oklahoma Form 511"},
    "Oregon": {"tax_code_name": "OR", "tax_form": "Oregon Form OR-40"},
    "Pennsylvania": {"tax_code_name": "PA", "tax_form": "Pennsylvania Form PA-40"},
    "Rhode Island": {"tax_code_name": "RI", "tax_form": "Rhode Island Form RI-1040"},
    "South Carolina": {"tax_code_name": "SC", "tax_form": "South Carolina Form SC1040"},
    "South Dakota": {"tax_code_name": "SD", "tax_form": "No state income tax"},
    "Tennessee": {"tax_code_name": "TN", "tax_form": "Tennessee Form INC-250"},
    "Texas": {"tax_code_name": "TX", "tax_form": "No state income tax"},
    "Utah": {"tax_code_name": "UT", "tax_form": "Utah Form TC-40"},
    "Vermont": {"tax_code_name": "VT", "tax_form": "Vermont Form IN-111"},
    "Virginia": {"tax_code_name": "VA", "tax_form": "Virginia Form 760"},
    "Washington": {"tax_code_name": "WA", "tax_form": "No state income tax"},
    "West Virginia": {"tax_code_name": "WV", "tax_form": "West Virginia Form IT-140"},
    "Wisconsin": {"tax_code_name": "WI", "tax_form": "Wisconsin Form 1"},
    "Wyoming": {"tax_code_name": "WY", "tax_form": "No state income tax"},
}

training = {
    "state": ["CA", "TX", "NY", "FL", "IL"],
    "hours_worked_per_week": [40, 35, 30, 45, 50],
    "hourly_rate": [50, 55, 60, 65, 70],
    "housing_cost": [1500, 1300, 1600, 1400, 1550],
    "meal_cost": [200, 180, 210, 190, 205],
    "transportation_cost": [100, 90, 110, 95, 105],
    "total_cost": [1800, 1570, 1920, 1685, 1860],
}


per_diem_rates = {
    "Alabama": {
        "Birmingham": {"Lodging": 126, "M&IE": 69},
        "Montgomery": {"Lodging": 98, "M&IE": 64},
        "Mobile": {"Lodging": 109, "M&IE": 64},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Alaska": {
        "Anchorage": {"Lodging": 241, "M&IE": 79},
        "Juneau": {"Lodging": 229, "M&IE": 79},
        "Fairbanks": {"Lodging": 202, "M&IE": 74},
        "Standard Rate": {"Lodging": 177, "M&IE": 74},
    },
    "Arizona": {
        "Phoenix": {"Lodging": 148, "M&IE": 69},
        "Tucson": {"Lodging": 137, "M&IE": 64},
        "Scottsdale": {"Lodging": 147, "M&IE": 69},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Arkansas": {
        "Little Rock": {"Lodging": 113, "M&IE": 64},
        "Fayetteville": {"Lodging": 112, "M&IE": 64},
        "Hot Springs": {"Lodging": 111, "M&IE": 64},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "California": {
        "Los Angeles": {"Lodging": 182, "M&IE": 74},
        "San Francisco": {"Lodging": 291, "M&IE": 79},
        "San Diego": {"Lodging": 177, "M&IE": 74},
        "Sacramento": {"Lodging": 147, "M&IE": 69},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Colorado": {
        "Denver": {"Lodging": 174, "M&IE": 74},
        "Colorado Springs": {"Lodging": 146, "M&IE": 69},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Connecticut": {
        "Hartford": {"Lodging": 147, "M&IE": 69},
        "New Haven": {"Lodging": 147, "M&IE": 69},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Delaware": {
        "Wilmington": {"Lodging": 155, "M&IE": 69},
        "Dover": {"Lodging": 114, "M&IE": 64},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Florida": {
        "Miami": {"Lodging": 206, "M&IE": 74},
        "Orlando": {"Lodging": 135, "M&IE": 69},
        "Tampa": {"Lodging": 140, "M&IE": 64},
        "Jacksonville": {"Lodging": 124, "M&IE": 64},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Georgia": {
        "Atlanta": {"Lodging": 163, "M&IE": 74},
        "Savannah": {"Lodging": 168, "M&IE": 69},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Hawaii": {
        "Honolulu": {"Lodging": 328, "M&IE": 97},
        "Maui": {"Lodging": 353, "M&IE": 94},
        "Standard Rate": {"Lodging": 177, "M&IE": 79},
    },
    "Idaho": {
        "Boise": {"Lodging": 121, "M&IE": 64},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Illinois": {
        "Chicago": {"Lodging": 246, "M&IE": 74},
        "Springfield": {"Lodging": 107, "M&IE": 59},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Indiana": {
        "Indianapolis": {"Lodging": 141, "M&IE": 69},
        "Fort Wayne": {"Lodging": 107, "M&IE": 59},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Iowa": {
        "Des Moines": {"Lodging": 117, "M&IE": 59},
        "Cedar Rapids": {"Lodging": 117, "M&IE": 59},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Kansas": {
        "Wichita": {"Lodging": 107, "M&IE": 59},
        "Topeka": {"Lodging": 107, "M&IE": 59},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Kentucky": {
        "Louisville": {"Lodging": 138, "M&IE": 69},
        "Lexington": {"Lodging": 122, "M&IE": 64},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Louisiana": {
        "New Orleans": {"Lodging": 176, "M&IE": 74},
        "Baton Rouge": {"Lodging": 107, "M&IE": 59},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Maine": {
        "Portland": {"Lodging": 149, "M&IE": 69},
        "Augusta": {"Lodging": 107, "M&IE": 59},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Maryland": {
        "Baltimore": {"Lodging": 157, "M&IE": 69},
        "Annapolis": {"Lodging": 149, "M&IE": 69},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Massachusetts": {
        "Boston": {"Lodging": 273, "M&IE": 74},
        "Cambridge": {"Lodging": 273, "M&IE": 74},
        "Worcester": {"Lodging": 107, "M&IE": 59},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Michigan": {
        "Detroit": {"Lodging": 145, "M&IE": 64},
        "Grand Rapids": {"Lodging": 117, "M&IE": 64},
        "Ann Arbor": {"Lodging": 134, "M&IE": 69},
        "Lansing": {"Lodging": 114, "M&IE": 64},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Minnesota": {
        "Minneapolis": {"Lodging": 161, "M&IE": 69},
        "Saint Paul": {"Lodging": 161, "M&IE": 69},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Mississippi": {
        "Jackson": {"Lodging": 107, "M&IE": 59},
        "Gulfport": {"Lodging": 129, "M&IE": 64},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Missouri": {
        "Kansas City": {"Lodging": 147, "M&IE": 69},
        "Saint Louis": {"Lodging": 151, "M&IE": 69},
        "Standard Rate": {"Lodging": 107, "M&IE": 59},
    },
    "Montana": {"Billings": {"Lodging": 128}},
}


def get_state_name(abbreviation):
    states = {
        "AL": "Alabama",
        "AK": "Alaska",
        "AZ": "Arizona",
        "AR": "Arkansas",
        "CA": "California",
        "CO": "Colorado",
        "CT": "Connecticut",
        "DE": "Delaware",
        "FL": "Florida",
        "GA": "Georgia",
        "HI": "Hawaii",
        "ID": "Idaho",
        "IL": "Illinois",
        "IN": "Indiana",
        "IA": "Iowa",
        "KS": "Kansas",
        "KY": "Kentucky",
        "LA": "Louisiana",
        "ME": "Maine",
        "MD": "Maryland",
        "MA": "Massachusetts",
        "MI": "Michigan",
        "MN": "Minnesota",
        "MS": "Mississippi",
        "MO": "Missouri",
        "MT": "Montana",
        "NE": "Nebraska",
        "NV": "Nevada",
        "NH": "New Hampshire",
        "NJ": "New Jersey",
        "NM": "New Mexico",
        "NY": "New York",
        "NC": "North Carolina",
        "ND": "North Dakota",
        "OH": "Ohio",
        "OK": "Oklahoma",
        "OR": "Oregon",
        "PA": "Pennsylvania",
        "RI": "Rhode Island",
        "SC": "South Carolina",
        "SD": "South Dakota",
        "TN": "Tennessee",
        "TX": "Texas",
        "UT": "Utah",
        "VT": "Vermont",
        "VA": "Virginia",
        "WA": "Washington",
        "WV": "West Virginia",
        "WI": "Wisconsin",
        "WY": "Wyoming",
    }

    return states.get(abbreviation.upper())


data = {
    "state": ["CA", "TX", "NY", "FL", "IL"],
    "hours_worked_per_week": [40, 35, 30, 45, 50],
    "hourly_rate": [50, 55, 60, 65, 70],
    "housing_cost": [1500, 1300, 1600, 1400, 1550],
    "meal_cost": [200, 180, 210, 190, 205],
    "transportation_cost": [100, 90, 110, 95, 105],
    "total_cost": [1800, 1570, 1920, 1685, 1860],
}


def get_state_abbreviation(state_name):
    states = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
    }

    return states.get(state_name.title())


def predict_contract_cost(data, inputInfo, weeks):
    df = pd.DataFrame(data)
    print(inputInfo["state"])
    if home_state == get_state_name(inputInfo["state"][0]):
        inputInfo["housing_cost"] = inputInfo["housing_cost"][0] - 500
        inputInfo["meal_cost"] = inputInfo["meal_cost"][0] - 50
        inputInfo["transportation_cost"] = inputInfo["transportation_cost"][0] - 50

    X = df.drop("total_cost", axis=1)
    y = df["total_cost"]

    preprocessor = ColumnTransformer(
        transformers=[("state", OneHotEncoder(handle_unknown="ignore"), ["state"])],
        remainder="passthrough",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )

    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    predicted_weekly_cost = model.predict(pd.DataFrame(inputInfo))[0]

    total_contract_cost = float(predicted_weekly_cost) * weeks

    return {
        "train_score": train_score,
        "test_score": test_score,
        "predicted_monthly_cost": predicted_weekly_cost,
        "total_contract_cost": total_contract_cost,
        "contract_weeks": weeks,
    }


def get_county_of_city(city, state=None, country=None):
    geolocator = Nominatim(user_agent="my_agent")

    try:
        # Construct the query string
        query = city
        if state:
            query += f", {state}"
        if country:
            query += f", {country}"

        # Perform the geocoding
        location = geolocator.geocode(query, addressdetails=True)

        if location and "address" in location.raw:
            address = location.raw["address"]
            # Check for various fields that might contain county information
            county = (
                address.get("county")
                or address.get("state_district")
                or address.get("district")
                or address.get("region")
            )

            if county:
                return county
            else:
                # If no county found, return all address details for inspection
                return f"County not found. Address details: {address}"
        else:
            return "Location not found"

    except (GeocoderTimedOut, GeocoderUnavailable):
        return "Geocoding service unavailable. Please try again later."


def find_tax_code(assignment):
    state_name = assignment["stateName"]
    if state_name in taxing:
        return (taxing[state_name]["tax_code_name"], taxing[state_name]["tax_form"])
    else:
        return "not found"


def find_per_diem(assignment):
    state_name = assignment["stateName"]
    city = assignment["city"]
    if state_name in per_diem_rates:
        if city in per_diem_rates[state_name]:
            return per_diem_rates[state_name][city]
        else:
            return per_diem_rates[state_name]
    else:
        return "not found"


def predict_and_project_costs(input_data, weeks):
    return predict_contract_cost(data, input_data, weeks)


def workIncome(assignment):
    getRate = float(assignment["rate"].strip('$'))
    if getRate != "Ask":
        # Extract the duration in weeks
        duration_str = (
            assignment["contractDuration"].strip().strip(" Weeks").strip(" weeks")
        )
        duration_in_weeks = int(duration_str)  # Convert to integer
        return (getRate * 40 * duration_in_weeks, duration_in_weeks)
    else:
        return (40000, 40000)


def calculate_benefits(assignment):
    print(type(workIncome(assignment)[0]))
    if workIncome(assignment)[0] > income:
        return "no-credit"
    else:
        credit = income - workIncome(assignment)[0]
        return credit

def find_assignment_by_id(job_data, target_id):
    for job in job_data:
        if job.get('id') == target_id:
            return job
    return None  # Return None if no matching job is found

def createInput(assignment):
    hourly_rate = (
        float(assignment["payRate"])
        if assignment.get("payRate")
        else float(assignment["rate"])
    )
    input_data = {
        "state": [
            get_state_abbreviation(assignment["stateName"])
        ],  # change this to abbreviation
        "hours_worked_per_week": [40],
        "hourly_rate": [hourly_rate],  # Convert rate to float
        "housing_cost": [
            1000
        ],  # Assuming housing cost is not provided in the original data
        "meal_cost": [400],  # Assuming meal cost is not provided in the original data
        "transportation_cost": [
            200
        ],  # Assuming transportation cost is not provided in the original data
    }
    return input_data


def createTax(assignment):
    inputData = createInput(assignment)
    tax_json = {
        "user-home-state": home_state,
        "traveling-state": assignment["stateName"],
        "home-state-income": income,
        "take-home-income": workIncome(assignment)[0],
        "projected-costs": predict_contract_cost(
            data, inputData, int(workIncome(assignment)[1] / 4)
        ),
        "per-diem-rate": find_per_diem(assignment),
        "credit-benefits": calculate_benefits(assignment),
        "tax-code": find_tax_code(assignment)[0],
        "tax-form": find_tax_code(assignment)[1],
        "double-expense-risk": assignment["double-expense"],
        "rate": assignment["rate"],
        "contract-duration": assignment["contractDuration"],
    }

    return tax_json


loader = PyPDFLoader("/Users/shauryasr/streamlit-keeper/p17.pdf")
pages = loader.load_and_split()

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings(api_key='sk-proj-OVkyeGEdBBwEMCWxRnDfT3BlbkFJMfpPIWGLmUtsqpjcdIi7'))
docs = faiss_index.similarity_search("How will the community be engaged?", k=2)
YDC_API_KEY = '65faebdf-a07e-4f26-a91d-089103fa1bdb<__>1PUkPOETU8N2v5f4roG3VKxL'
api_wrapper = YouSearchAPIWrapper(ydc_api_key=YDC_API_KEY,num_web_results = 10)
ydc_tool = YouSearchTool(api_wrapper=api_wrapper)
llm = ChatAnthropic(api_key='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', model='claude-3-5-sonnet-20240620')
# Create a checkpointer to use memory
memory = MemorySaver()

# convert this retriver into a tool
db_retriever = faiss_index.as_retriever()
db_retriever_tool = create_retriever_tool(
    db_retriever,
    name = "taw_dataset_retriever",
    description = "Retrieve relevant context from the US tax dataset."
)
# the vector store representation of the CSV dataset and the You.com Search tool will both be passed as tools to the agent
tools = [db_retriever_tool, ydc_tool]
agent_executor = chat_agent_executor.create_tool_calling_executor(llm, tools, checkpointer=memory)


def create_overlay_pdf():
    #this will define the ELEMENTS that will compose the template. 
    elements = [
        #{ 'name': 'company_logo', 'type': 'I', 'x1': 20.0, 'y1': 17.0, 'x2': 78.0, 'y2': 30.0, 'font': None, 'size': 0.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': 'logo', 'priority': 2, },
        { 'name': 'first_name', 'type': 'T', 'x1': 20.0, 'y1': 81.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'last_name', 'type': 'T', 'x1': 85.0, 'y1': 81.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'M_initial', 'type': 'T', 'x1': 63.5, 'y1': 81.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'suffix', 'type': 'T', 'x1': 140.0, 'y1': 81.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 's_s_n', 'type': 'T', 'x1': 160.0, 'y1': 81.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'spouse_fname', 'type': 'T', 'x1': 20.0, 'y1': 101.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'spouse_initial', 'type': 'T', 'x1': 63.5, 'y1': 101.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'spouse_lname', 'type': 'T', 'x1': 85.0, 'y1': 101.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'spouse_suffix', 'type': 'T', 'x1': 140.0, 'y1': 101.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'spouse_ssn', 'type': 'T', 'x1': 160.0, 'y1': 101.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'additional_info', 'type': 'T', 'x1': 20.0, 'y1': 121.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'PBA_code', 'type': 'T', 'x1': 20.0, 'y1': 34.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'street_address', 'type': 'T', 'x1': 20.0, 'y1': 141.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'apt_no', 'type': 'T', 'x1': 140.0, 'y1': 141.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'priv_mailbx', 'type': 'T', 'x1': 20.0, 'y1': 34.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'city', 'type': 'T', 'x1': 20.0, 'y1': 161.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'state', 'type': 'T', 'x1': 142.0, 'y1': 160.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'zip_code', 'type': 'T', 'x1': 160.0, 'y1': 160.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'dob', 'type': 'T', 'x1': 26.0, 'y1': 213.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'foreign_country_name', 'type': 'T', 'x1': 30.0, 'y1': 180.5, 'x2': 115.0, 'y2': 37.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'if_single', 'type': 'T', 'x1': 27.0, 'y1': 185.5, 'x2': 115.0, 'y2': 217.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'if_marriedFS', 'type': 'T', 'x1': 27.0, 'y1': 203.5, 'x2': 115.0, 'y2': 217.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'if_marriedFJ', 'type': 'T', 'x1': 27.0, 'y1': 236.5, 'x2': 115.0, 'y2': 217.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'if_head_of_household', 'type': 'T', 'x1': 27.0, 'y1': 185.5, 'x2': 115.0, 'y2': 217.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },
        { 'name': 'if_surviving_spouse', 'type': 'T', 'x1': 27.0, 'y1': 185.5, 'x2': 115.0, 'y2': 217.5, 'font': 'Arial', 'size': 11.0, 'bold': 0, 'italic': 0, 'underline': 0, 'foreground': 0, 'background': 0, 'align': 'I', 'text': '', 'priority': 2, },



    ]

    #here we instantiate the template and define the HEADER
    f = Template(format="A4", elements=elements,title="Sample Invoice")
    f.add_page()

    #we FILL some of the fields of the template with the information we want
    #note we access the elements treating the template instance as a "dict"
    f["first_name"] = "Shaurya"
    f["last_name"] = "Srivastava"
    f["M_initial"] = "N"
    f["suffix"] = "III"
    f["s_s_n"] = "234234243"
    f["spouse_fname"] = "Mili"
    f["spouse_initial"] = "V"
    f["spouse_lname"] = "Patel"
    f["spouse_suffix"] = "IV"
    f["spouse_ssn"] = "6969696969"
    f["additional_info"] = "Dual Citizen (Canada)"
    f["street_address"] = '3 Cedar St'
    f['city'] = 'Princeton Junction'
    f['state'] = 'NJ'
    f['zip_code'] = '08550'
    f['apt_no'] = "3c"
    f['dob'] = '3/22/2004'
    f['if_single'] = 'x'
    f['if_marriedFS'] = 'x'
    f['if_marriedFJ'] = 'x'
    f['if_head_of_household'] = 'x'
    f['if_surviving_spouse'] = 'x'


    #and now we render the page
    f.render("Cali_income_tax_return.pdf")

def merge_pdfs(base_pdf_path, overlay_pdf_path, output_pdf_path):
    base_pdf = PdfReader(base_pdf_path)
    overlay_pdf = PdfReader(overlay_pdf_path)
    
    for page_num, base_page in enumerate(base_pdf.pages):
        if page_num < len(overlay_pdf.pages):
            overlay = PageMerge().add(overlay_pdf.pages[page_num])[0]
            PageMerge(base_page).add(overlay).render()
    
    PdfWriter(output_pdf_path, trailer=base_pdf).write()


def main():
    job_data = fetch_and_process_jobs(
        url="https://www.geniehealthjobs.com/api/jobs/search?cacheBuster=1718420686404&page=1&query=%7B%22id%22:null,%22city%22:%22%22,%22state%22:%5B%5D,%22profession%22:null,%22specialty%22:null,%22shift%22:null,%22payRate%22:null,%22startDate%22:null,%22endDate%22:null,%22modifiedDate%22:null,%22latestCreatedDate%22:null,%22empType%22:%5B%5D,%22noOfPositions%22:%221%2B%22,%22contractDuration%22:null,%22jobDesc%22:null,%22rate%22:null,%22facility%22:null,%22jobSource%22:null,%22sourceJobID%22:null,%22accountManagerName%22:null,%22lastModifiedDate%22:null,%22stateArray%22:%5B%5D%7D&size=8000&sort=id,asc"
    )
    assignmentJob = find_assignment_by_id(job_data=job_data, target_id=assignment_id)
    returnedOutput = createTax(assignment=assignmentJob)
    st.write(returnedOutput)
    prompt = st.text_input("Enter your prompt, regarding any question on taxes and our model will answer it:")
    if prompt:
        st.write(agent_executor.invoke(input={"messages": prompt}, config={"configurable": {"thread_id": "xyz_789"}})["messages"][-1].content)
    
    create_overlay_pdf()
    
    base_pdf_path = '/Users/shauryasr/streamlit-keeper/Cali_tax.pdf'
    overlay_pdf_path = '/Users/shauryasr/streamlit-keeper/Cali_income_tax_return.pdf'
    output_pdf_path = '/Users/shauryasr/streamlit-keeper/Final_Output.pdf'
    merge_pdfs(base_pdf_path, overlay_pdf_path, output_pdf_path)


if __name__ == "__main__":
    main()