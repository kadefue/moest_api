# MOEST Enrollment Forecasting API

This project implements a Machine Learning-based forecasting system for Secondary School enrollment in Tanzania. It uses FastAPI for the REST interface and a combination of XGBoost, LightGBM, and other regression models to predict student enrollment numbers.

## Features

- **Automated ML Pipeline**: Data cleaning, feature engineering, and model training.
- **Ensemble Modeling**: Uses XGBoost, LightGBM, Random Forest, and Gradient Boosting.
- **REST API**: FastAPI endpoints to trigger forecasts and retrieve results.
- **Dockerized**: specific container setup for reproducible environments.
- **MySQL Integration**: Stores forecast results in a MySQL database.

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Installation & Running

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd moest_api
    ```

2.  **Prepare the Data**:
    *   Create the data directory structure if it doesn't exist:
        ```bash
        mkdir -p data/csvs
        ```
    *   **CRITICAL**: Place your BEST dataset CSV files into `data/csvs/`.

3.  **Run with Docker Compose**:
    ```bash
    docker-compose up --build
    ```
    This command will:
    - Start the MySQL database container.
    - Build the Python application container.
    - Start the FastAPI server on port `8000`.

4.  **Access the API**:
    - **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
    - **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Data Requirements

The system currently ingests data from CSV files located in the `data/csvs/` directory.

> **Note**: In future versions, this system will directly query a centralized Database or DatasetAPI. For now, it relies on the manual upload of the BEST dataset CSVs.

### Required File Structure

Ensure the following CSV files (from the BEST dataset) are present in `data/csvs/`:

- `Secondary_enrol_all_2016_2025.csv` (Aggregate enrollment)
- `Secondary_enrol_Gov_2016_2025.csv` (Government enrollment)
- `Secondary_students_per_subject.csv` (Detailed subject enrollment)
- `Data-Secondary Tables and chairs 2016-2025.csv`
- `Dropout-Secondary 2017-2024.csv`
- `Secondary-Re_entry.csv`
- `Secondary - DISABALITY 2020-2025.csv`
- `Combined_Secondary_ICT_All_G_NG.csv`
- `Combined_Secondary_Electricity_All_G_NG.csv`
- `Combined_Secondary_Laboratories_All_G_NG.csv`
- `LGAs Urban and Rural Status.csv` (For urban/rural classification)

**Geodata:**
- Place `tanzania_council_geodata.csv` in the `data/` root (or `data/csvs/` depending on your mount, but the default config looks in `/app/data/`). The `docker-compose.yml` mounts `./data` to `/app/data`, so putting it in `./data/tanzania_council_geodata.csv` is recommended.

### CSV Formatting
- **Headers**: Files should generally include `Year`, `Region`, and `Council` columns.
- **Cleaning**: The pipeline automatically handles common formatting issues (e.g., removing commas from numbers, handling Roman numerals for Forms).

## Usage

### Triggering a Forecast
To run the full training and forecasting pipeline:

```bash
curl -X POST "http://localhost:8000/run-forecast"
```

This will:
1. Load CSVs.
2. Train models on data pre-2025.
3. Validate against 2025 data.
4. Generate forecasts for 2026-2030.
5. Save results to the MySQL database.

### Viewing Results
You can query the results via the API:

```bash
curl "http://localhost:8000/forecasts?year=2026&region=Arusha"
```