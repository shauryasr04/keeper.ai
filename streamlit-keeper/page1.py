import streamlit as st
import pandas as pd
from groq import Groq
import requests
import re

# Set page configuration
st.set_page_config(page_title="Keeper - Job Search", layout="wide")
st.sidebar.header("New Demo")
st.sidebar.success("Select Pages")
# Custom CSS styles
st.markdown(
    """
<style>
.card {
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    border-radius: 5px;
    padding: 20px;
    margin: 10px;
    background-color: white;
    font-color: black;
}
.card h3{
    color: black;
    margin-bottom: 10px;
}
.card:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}
a.card-link{
    text-decoration: none;
    color: inherit;
    display: block;
}
a.card-link:hover{
    text-decoration: none;
    color:inherit:
}
</style>
""",
    unsafe_allow_html=True,
)


def covert_to_json_groq(input):
    # Regex pattern to extract job details
    job_pattern = re.compile(
        r"Job Title: (.*?)\s+Location: (.*?)\s+Facility: (.*?)\s+Specialty: (.*?)\s+Job Type: (.*?)\s+Start Date: (.*?)\s+End Date: (.*?)\s+Contract Duration: (.*?)\s+Rate: (.*?)\s+Job Application Link: (.*?)\s+"
    )

    # Find all matches
    matches = job_pattern.findall(input)

    # List to hold job dictionaries
    jobs = []

    # Populate the jobs list with formatted job data
    for i, match in enumerate(matches):
        job = {
            "id": match[9].strip().strip("https://www.geniehealthjobs.com/#/job/"),
            "title": match[0].strip(),
            "company": match[2].strip(),
            "type": f"Job Type: {match[4].strip()}",
            "location": f"Location: {match[1].strip()}",
            "specialty": f"Specialty: {match[3].strip()}",
            "link": f" {match[9].strip()}",
            "contract": f"Contract Duration: {match[7].strip()}",
            "rate": f"Rate: {match[8].strip()}"
            
        }
        jobs.append(job)

    return jobs


def groqify(res):
    count = 0
    results = []
    client = Groq(
        api_key="",
    )
    for i, content in enumerate(res):
        chunk = res[i : i + 2]
        if count <= 10:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"You are a nursing recruiting agent that helps nurses find their dream job, based on the input that you are given {chunk}. Return the top two job postings within the json input that have the most content. Print it out in the following manner, with each element in a new line and each job as a seperate output: Job Title, Location, Facility, Specialty, Job Type, Start Date, End Date, Contract Duration, Rate, and Job Application Link (Include the link). If you can't find anything related to a category, just write N/A. Only include this in your output. Please don't write anything else but my requirements. ",
                    }
                ],
                model="llama3-8b-8192",
            )
            results.append(chat_completion.choices[0].message.content)
        count += 2
    return " ".join(results)


# Function to create job cards
def create_job_card(job):
    with st.container():
        st.markdown(
            f"""
        <a href = "{job['link']}" class = "card-link" target = "_blank">
            <div class="card">
                <h3>{job['id']}</h3>
                <h3>{job['title']}</h3>
                <h3>{job['company']}</h3>
                <h3>{job['type']}</h3>
                <h3>{job['location']}</h3>
                <h3>{job['specialty']}</h3>
                <h3>{job['contract']}</h3>
                <h3>{job['rate']}</h3>
                <button>Tax Information<button>
            </div>
        </a>
        """,
            unsafe_allow_html=True,
        )


# Function to fetch and process jobs
def fetch_and_process_jobs(url):
    # Send GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON content
        json_data = response.json()

        # Iterate through each element in the JSON data
        for element in json_data:
            id = element["id"]
            link = f"https://www.geniehealthjobs.com/#/job/{id}"

            # Add a new key-value pair to each element
            element["job_apply"] = link

            # Check contract duration condition
            contract_duration = element.get("contractDuration", "")
            if contract_duration and contract_duration != "Ask":
                # Handle variations in case for " weeks"
                if "weeks" in contract_duration.lower():
                    duration_in_weeks = int(
                        contract_duration.strip(" weeks").strip().strip("WEEKS")
                    )
                else:
                    duration_in_weeks = int(contract_duration)

                if duration_in_weeks >= 26:
                    element["double-expense"] = True
                else:
                    element["double-expense"] = False
            else:
                element["double-expense"] = False

        return json_data
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None


# Function to filter jobs
def filter_jobs(jobs, state_name, profession, emp_type):
    filtered_jobs = [
        job
        for job in jobs
        if (job["stateName"] and job["stateName"].lower() == state_name.lower())
        or (
            job["directStateName"]
            and job["directStateName"].lower() == state_name.lower()
        )
        and job["profession"].lower() == profession.lower()
        and job["empType"].lower() == emp_type.lower()
    ]
    return filtered_jobs


# Function to display job listings
def display_jobs(jobs):
    st.title("Job Listings")
    if jobs:
        for job in jobs:
            create_job_card(job)
    else:
        st.warning("No jobs found matching the selected criteria.")


# Main function
def main():
    st.title("Welcome to Keeper")
    st.subheader(
        "Find the perfect job matching your skills and preferences across all 50 states."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.selectbox(
            "State",
            [
                "Alabama",
                "Alaska",
                "Arizona",
                "Arkansas",
                "California",
                "Colorado",
                "Connecticut",
                "Delaware",
                "Florida",
                "Georgia",
                "Hawaii",
                "Idaho",
                "Illinois",
                "Indiana",
                "Iowa",
                "Kansas",
                "Kentucky",
                "Louisiana",
                "Maine",
                "Maryland",
                "Massachusetts",
                "Michigan",
                "Minnesota",
                "Mississippi",
                "Missouri",
                "Montana",
                "Nebraska",
                "Nevada",
                "New Hampshire",
                "New Jersey",
                "New Mexico",
                "New York",
                "North Carolina",
                "North Dakota",
                "Ohio",
                "Oklahoma",
                "Oregon",
                "Pennsylvania",
                "Rhode Island",
                "South Carolina",
                "South Dakota",
                "Tennessee",
                "Texas",
                "Utah",
                "Vermont",
                "Virginia",
                "Washington",
                "West Virginia",
                "Wisconsin",
                "Wyoming",
            ],
        )

    with col2:
        employer_type = st.selectbox("Employer Type", ["Travel", "Per Diem", "Locum"])

    with col3:
        profession = st.selectbox(
            "Profession",
            [
                "Administrative",
                "Adult Nurse Practitioner",
                "Allied Health Professional",
                "Cardio Pulmonary",
                "Therapy/Rehabilitation",
            ],
        )

    job_data = fetch_and_process_jobs(
        url="https://www.geniehealthjobs.com/api/jobs/search?cacheBuster=1718420686404&page=1&query=%7B%22id%22:null,%22city%22:%22%22,%22state%22:%5B%5D,%22profession%22:null,%22specialty%22:null,%22shift%22:null,%22payRate%22:null,%22startDate%22:null,%22endDate%22:null,%22modifiedDate%22:null,%22latestCreatedDate%22:null,%22empType%22:%5B%5D,%22noOfPositions%22:%221%2B%22,%22contractDuration%22:null,%22jobDesc%22:null,%22rate%22:null,%22facility%22:null,%22jobSource%22:null,%22sourceJobID%22:null,%22accountManagerName%22:null,%22lastModifiedDate%22:null,%22stateArray%22:%5B%5D%7D&size=8000&sort=id,asc"
    )

    if job_data:
        filtered_jobs = filter_jobs(job_data, state, profession, employer_type)
        resultant_output = groqify(filtered_jobs)
        newJson = covert_to_json_groq(resultant_output)
        display_jobs(newJson)
        if "my_input" not in st.session_state:
            st.session_state["my_input"] = ""
        my_input = st.text_input("Choose your desired Job ID, home state, and income for home state for your relevant tax information - eg: (jobid, home-state, income): ", st.session_state["my_input"])
        submit = st.button("Input Submit")
        if submit and my_input :
            st.session_state["my_input"] = my_input
        else:
            st.warning("Please enter a valid job id!")




        #exit
if __name__ == "__main__":
    main()
