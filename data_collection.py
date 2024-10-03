import os
import requests
import zipfile
import pandas as pd

# O*NET data download URL (no API key needed)
ONET_DATA_URL = "https://www.onetcenter.org/dl_files/database/db_24_2_text.zip"

# Directory for caching the data
CACHE_DIR = "./"

def download_onet_data():
    """
    Downloads the O*NET database ZIP file and extracts it to the cache directory.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    zip_file_path = os.path.join(CACHE_DIR, 'onet_data.zip')

    # Check if the data is already downloaded
    if not os.path.exists(zip_file_path):
        print("Downloading O*NET data...")
        response = requests.get(ONET_DATA_URL, stream=True)
        if response.status_code == 200:
            with open(zip_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)
            print("Download completed.")
        else:
            raise Exception(f"Failed to download O*NET data (status code: {response.status_code})")
    else:
        print("O*NET data already downloaded.")

    # Extract the zip file
    extract_onet_data(zip_file_path)


def extract_onet_data(zip_file_path):
    """
    Extracts the downloaded O*NET data zip file into the cache directory.
    """
    print("Extracting O*NET data...")

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(CACHE_DIR)

    print("Extraction completed.")


def load_onet_csv(filename):
    """
    Loads a specific O*NET CSV file from the current working directory.

    Args:
        filename (str): Name of the CSV file to load (e.g., "Skills.txt", "WorkActivities.txt").

    Returns:
        pd.DataFrame: The loaded DataFrame from the CSV file.
    """
    file_path = os.path.join(CACHE_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{filename} not found in the current directory at {file_path}.")

    # O*NET CSVs are tab-delimited, so we set sep="\t"
    return pd.read_csv(file_path, sep="\t")


def collect_existing_industry_data():
    """
    Collects data for existing industries by parsing the relevant O*NET CSV files
    (skills, work activities) and returns them as a pandas DataFrame.

    Returns:
        pd.DataFrame: Data for existing industries including skills and work activities.
    """
    # Download and cache O*NET data if not already done
    download_onet_data()

    # Load relevant CSVs
    skills_df = load_onet_csv("Skills.txt")
    work_activities_df = load_onet_csv("Work Activities.txt")

    # Example: Join and filter the data (customize as needed)
    # Merge skills and work activities by their occupation codes (SOC Code)
    industry_data = pd.merge(skills_df, work_activities_df, on='O*NET-SOC Code', how='inner')

    # Filter relevant columns for simplicity (customize based on actual needs)
    industry_data = industry_data[['O*NET-SOC Code', 'Element Name_x', 'Element Name_y']]
    industry_data.columns = ['SOC Code', 'Skills', 'Work Activities']

    return industry_data


def collect_target_industry_data(target_industry_soc_code):
    """
    Collects data for the target industry by filtering the relevant O*NET CSV files
    based on the SOC code of the target industry.

    Args:
        target_industry_soc_code (str): The SOC code of the target industry (e.g., "15-1132.00").

    Returns:
        dict: Target industry data (skills and work activities).
    """
    # Download and cache O*NET data if not already done
    download_onet_data()

    # Load relevant CSVs
    skills_df = load_onet_csv("Skills.txt")
    work_activities_df = load_onet_csv("Work Activities.txt")

    # Filter data for the target industry using SOC code
    target_skills = skills_df[skills_df['O*NET-SOC Code'] == target_industry_soc_code]
    target_work_activities = work_activities_df[work_activities_df['O*NET-SOC Code'] == target_industry_soc_code]

    # Combine the relevant data into a single dictionary
    target_data = {
        'SOC Code': target_industry_soc_code,
        'Skills': target_skills['Element Name'].tolist(),
        'Work Activities': target_work_activities['Element Name'].tolist(),
    }

    return target_data


if __name__ == "__main__":
    # Example usage:

    # Collect data for existing industries
    existing_industries = collect_existing_industry_data()
    print("Existing Industry Data:")
    print(existing_industries.head())  # Print a preview

    # Collect data for a specific target industry (example: SOC code for software developers)
    target_industry_soc_code = '15-1132.00'  # Example: Software Developers
    target_industry_data = collect_target_industry_data(target_industry_soc_code)

    print("\nTarget Industry Data:")
    print(target_industry_data)
