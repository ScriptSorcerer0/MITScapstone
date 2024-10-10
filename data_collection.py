import os
import requests
import zipfile
import pandas as pd

ONET_DATA_URL = "https://www.onetcenter.org/dl_files/database/db_24_2_text.zip"
WITS_API_URL = "https://wits.worldbank.org/API/V1/SDMX/V21/datasource/TOTAL"
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
CACHE_DIR = "./"


def download_onet_data():
    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_file_path = os.path.join(CACHE_DIR, 'onet_data.zip')
    if not os.path.exists(zip_file_path):
        response = requests.get(ONET_DATA_URL, stream=True)
        if response.status_code == 200:
            with open(zip_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)
    extract_onet_data(zip_file_path)


def extract_onet_data(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(CACHE_DIR)


def load_onet_csv(filename):
    file_path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{filename} not found in {file_path}.")
    return pd.read_csv(file_path, sep="\t")


def collect_existing_industry_data():
    download_onet_data()

    skills_df = load_onet_csv("Skills.txt")
    tasks_to_dwa_df = load_onet_csv("Tasks to DWAs.txt")
    dwa_reference_df = load_onet_csv("DWA Reference.txt")

    # Merge tasks with DWAs, and DWAs with their reference information
    tasks_with_dwa = pd.merge(tasks_to_dwa_df, dwa_reference_df, on="DWA ID", how="inner")

    # Merge the resulting DWAs with skills using SOC code
    industry_data = pd.merge(skills_df, tasks_with_dwa, on='O*NET-SOC Code', how='inner')

    # Adjust the column selection based on the available columns in the merged data
    industry_data = industry_data[['O*NET-SOC Code', 'Element Name', 'DWA Title', 'Task ID']]

    # Renaming the columns
    industry_data.columns = ['SOC Code', 'Skills', 'DWA Title', 'Task ID']

    # Trim to 6-digit SOC Code
    industry_data['SOC Code'] = industry_data['SOC Code'].str[:6]

    return industry_data


def collect_bls_workers_data(series_id):
    url = f"{BLS_API_URL}{series_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['Results']['series'][0]['data'][0]['value']  # Return the number of workers
    return None


def collect_target_industry_data(target_industry_soc_code):
    download_onet_data()

    skills_df = load_onet_csv("Skills.txt")
    tasks_to_dwa_df = load_onet_csv("Tasks to DWAs.txt")
    dwa_reference_df = load_onet_csv("DWA Reference.txt")

    tasks_with_dwa = pd.merge(tasks_to_dwa_df, dwa_reference_df, on="DWA ID", how="inner")
    target_skills = skills_df[skills_df['O*NET-SOC Code'].str.startswith(target_industry_soc_code[:6])]
    target_tasks = tasks_with_dwa[tasks_with_dwa['O*NET-SOC Code'].str.startswith(target_industry_soc_code[:6])]

    target_data = {
        'SOC Code': target_industry_soc_code,
        'Skills': target_skills['Element Name'].tolist(),
        'DWA Titles': target_tasks['DWA Title'].tolist(),
        'Tasks ID': target_tasks['Task ID'].tolist(),
    }

    return target_data


def collect_wits_trade_data(product_code, country_code):
    params = {
        "product": product_code,
        "country": country_code,
        "year": "2020",
        "format": "json"
    }
    response = requests.get(WITS_API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    return None


if __name__ == "__main__":
    existing_industries = collect_existing_industry_data()
    print(existing_industries.head())

    target_industry_soc_code = '15-1132.00'
    target_industry_data = collect_target_industry_data(target_industry_soc_code)
    print(target_industry_data)

    steel_trade_data = collect_wits_trade_data("7207", "USA")
    print(steel_trade_data)

    bls_workers = collect_bls_workers_data("LAUCN040010000000005")
    print(f"Number of workers: {bls_workers}")
