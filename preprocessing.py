import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from data_collection import collect_existing_industry_data, collect_target_industry_data
from scipy.sparse import hstack


# Directory for saving processed data
PROCESSED_DIR = "./"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_data(data):
    for i in range(data.shape[1]):
        if np.issubdtype(data[:, i].dtype, np.str_):
            data[:, i] = np.char.strip(data[:, i])
    return data


def preprocess_in_batches(raw_data, batch_size=100000):
    num_rows = raw_data.shape[0]

    for start_idx in range(0, num_rows, batch_size):
        end_idx = min(start_idx + batch_size, num_rows)

        batch_data = raw_data[start_idx:end_idx]

        cleaned_data = clean_data(batch_data)

        skills_col_idx = 1
        dwa_title_col_idx = 2
        soc_code_col_idx = 0

        remaining_data, encoded_data = one_hot_encode(cleaned_data, [skills_col_idx, dwa_title_col_idx])

        remaining_data = np.array(remaining_data)


        normalized_data = normalize_columns(remaining_data, [soc_code_col_idx])

        batch_output = np.hstack((normalized_data, encoded_data.toarray()))

        output_path = os.path.join(PROCESSED_DIR, "preprocessed_existing_industries_batch.csv")
        with open(output_path, 'a') as f:
            np.savetxt(f, batch_output, delimiter=",", fmt="%s")

    print("Batch processing completed.")


def one_hot_encode(data, columns_indices):
    encoder = OneHotEncoder(sparse=True)  # Keep sparse format to save memory
    encoded = encoder.fit_transform(data[:, columns_indices])

    remaining_data = np.delete(data, columns_indices, axis=1)

    return remaining_data, encoded



def normalize_columns(data, columns_indices):
    for col_idx in columns_indices:
        try:
            data[:, col_idx] = data[:, col_idx].astype(float)
            scaler = StandardScaler()
            data[:, col_idx] = scaler.fit_transform(data[:, col_idx].reshape(-1, 1)).flatten()
        except ValueError:
            print("skip")

    return data





def preprocess_existing_industry_data():
    raw_data = collect_existing_industry_data().to_numpy()

    if raw_data.size == 0:
        raise ValueError("No data was loaded. Please check if the data collection is working correctly.")

    print("Raw data shape:", raw_data.shape)

    cleaned_data = clean_data(raw_data)

    skills_col_idx = 1  # 'Skills' column
    dwa_title_col_idx = 2  # 'DWA Title' column
    soc_code_col_idx = 0  # 'SOC Code' column

    remaining_data, encoded_data = one_hot_encode(cleaned_data, [skills_col_idx, dwa_title_col_idx])

    normalized_remaining_data = normalize_columns(remaining_data, [soc_code_col_idx])

    batch_output = np.hstack((normalized_remaining_data, encoded_data.toarray()))

    np.savetxt(os.path.join(PROCESSED_DIR, "preprocessed_existing_industries.csv"), batch_output, delimiter=",", fmt="%s")

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
