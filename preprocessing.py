import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from data_collection import collect_existing_industry_data, collect_target_industry_data  # Import from data_collection

# Directory for saving processed data
PROCESSED_DIR = "./"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_data(data):
    data[data == ''] = 'Unknown'
    return np.char.strip(data)


def one_hot_encode(data, columns_indices):
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(data[:, columns_indices])
    remaining_data = np.delete(data, columns_indices, axis=1)
    return np.hstack((remaining_data, encoded))


def normalize_columns(data, columns_indices):
    scaler = StandardScaler()
    data[:, columns_indices] = scaler.fit_transform(data[:, columns_indices].astype(float))
    return data


def preprocess_existing_industry_data():
    raw_data = collect_existing_industry_data().to_numpy()

    cleaned_data = clean_data(raw_data)

    skills_col_idx = np.where(raw_data[0] == 'Skills')[0][0]
    dwa_title_col_idx = np.where(raw_data[0] == 'DWA Title')[0][0]
    soc_code_col_idx = np.where(raw_data[0] == 'SOC Code')[0][0]

    encoded_data = one_hot_encode(cleaned_data, [skills_col_idx, dwa_title_col_idx])
    normalized_data = normalize_columns(encoded_data, [soc_code_col_idx])

    # Save preprocessed data
    np.savetxt(os.path.join(PROCESSED_DIR, "preprocessed_existing_industries.csv"), normalized_data, delimiter=",",
               fmt="%s")


def preprocess_target_industry_data():
    raw_data = collect_target_industry_data('15-1132.00')  # Example SOC Code
    raw_data = np.array(list(raw_data.values()))  # Convert dict to NumPy array

    cleaned_data = clean_data(raw_data)

    encoded_data = one_hot_encode(cleaned_data, [1, 2])  # For Skills and DWA Titles
    normalized_data = normalize_columns(encoded_data, [0])  # For SOC Code

    np.savetxt(os.path.join(PROCESSED_DIR, "preprocessed_target_industry.csv"), normalized_data, delimiter=",",
               fmt="%s")


if __name__ == "__main__":
    preprocess_existing_industry_data()
    preprocess_target_industry_data()
