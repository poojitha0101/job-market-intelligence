# India Data & AI Job Market Intelligence

An end-to-end data analytics project that explores job listings related to Data and AI in India using Python, SQL, Power BI, and a Retrieval-Augmented Generation (RAG) based AI assistant.

## Project Overview

Finding useful information in job listings can take time. This project explores a sample of job listings to understand the roles, companies, locations, and skills mentioned in job descriptions.

The project combines a Power BI dashboard for exploring overall trends with an AI assistant that answers questions using relevant job listings.

The data was collected through the Adzuna API.

## Key Highlights

- Collected 800 raw job records using the Adzuna API.
- Cleaned the data and removed duplicate job IDs, leaving 784 unique job listings.
- Used Python and SQLite for data preparation and analysis.
- Built an interactive Power BI dashboard with charts and slicers.
- Developed an AI assistant using embeddings, RAG, and an LLM.
- Deployed the AI assistant using Streamlit.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data processing and application development |
| Pandas | Data cleaning and manipulation |
| NumPy | Data processing |
| SQLite | Storing and querying job data |
| SQL | Analysing job roles, companies, locations, and skills |
| Power BI | Interactive dashboard and visualisation |
| Embeddings | Representing job listings as vectors |
| RAG | Retrieving relevant listings before generating answers |
| LLM | Generating answers from retrieved listings |
| Streamlit | Building and deploying the AI assistant |
| Git and GitHub | Version control and project hosting |

## Project Workflow

### 1. Data Collection

Collected 800 job records from the Adzuna API for analysis.

### 2. Data Cleaning

Used Python, Pandas, and NumPy to prepare the data and remove duplicate job IDs. After cleaning, 784 unique job listings remained.

### 3. Data Storage

Stored the prepared job data in a SQLite database for querying and analysis.

### 4. SQL Analysis

Used SQL queries to explore:

- Job roles and listing counts
- Companies appearing in the collected listings
- Locations mentioned in the data
- Skills mentioned in job descriptions

### 5. Power BI Dashboard

Created an interactive dashboard to explore the collected job listings.

The dashboard includes charts and slicers for exploring:

- Job roles
- Companies
- Job locations
- Skills mentioned in job descriptions

Users can apply filters to explore different parts of the dataset.

### 6. AI Job Assistant

Built an AI assistant that allows users to ask questions about the collected job listings in natural language.

The assistant uses embeddings to represent job listings and semantic search to find listings relevant to a user's question.

The retrieved listings are passed to an LLM through a RAG workflow to generate an answer based on the available information.

Example questions:

- Which jobs need both Python and SQL?
- Which jobs mention LLM or RAG?

The assistant returns relevant answers with job references, such as `[Job 1]` and `[Job 2]`, so users can check the matching listings.

## Dashboard vs AI Assistant

**Power BI Dashboard**

The dashboard is useful for comparing and exploring overall trends across the collected listings, including roles, companies, locations, and skills.

**AI Job Assistant**

The assistant is designed to answer questions about individual job listings by retrieving relevant matches from the collected data.

Both components serve different purposes and are part of the same project.

## Project Links

- **Live AI Assistant:** https://india-job-market-assistant.streamlit.app/
- **Project Overview PDF:** https://drive.google.com/file/d/1emgFfmqUzvZY93wY0g4QioUP4azlbtCj/view?usp=sharing
- **GitHub Repository:** https://github.com/poojitha0101/job-market-intelligence

## Running the Project Locally

### Requirements

- Python
- Git
- The project dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/poojitha0101/job-market-intelligence.git
   ```

2. Move into the project directory:

   ```bash
   cd job-market-intelligence
   ```

3. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate the environment.

   **Windows:**

   ```bash
   .venv\Scripts\activate
   ```

5. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Follow the application's entry-point and configuration instructions in the `app` and `.streamlit` folders to start the assistant.

   The exact Streamlit command depends on the application's main Python file.

### API Configuration

The deployed assistant asks users to provide their own free LLM API key. The key is used for the session and is not saved by the application, as described in the project overview.

Never commit API keys, passwords, or other secrets to GitHub.

## Limitations

- This project uses a sample of 784 unique Adzuna job listings. It does not represent the entire Indian job market.
- Skill counts are based on keyword matching, so some mentions may be missed.
- The AI assistant retrieves only the five closest matching listings for a question.
- The assistant is not designed to count or rank all 784 listings. The Power BI dashboard is better suited for exploring overall counts and trends.
- Some job descriptions are truncated, and salary information is missing for many listings.

## What I Learned

Through this project, I practised data collection, data cleaning, SQL analysis, dashboard development, semantic search, and integrating an LLM with retrieved data.

It also helped me understand how traditional data analytics and AI-based question answering can work together in one application.

## Author

**Poojitha Nallamaru**

This is a personal portfolio project built for learning and demonstrating practical skills in data analytics and AI.
