import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

# Filepaths for saving preprocessed data
PROCESSED_DIR = "./processed_data"
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)


def clean_data(df):
    """
    Cleans the input DataFrame by handling missing values and ensuring consistent formatting.

    Args:
        df (pd.DataFrame): Raw DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Fill missing values with a placeholder (customize based on your needs)
    df.fillna('Unknown', inplace=True)

    # Strip leading/trailing spaces from string columns
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].str.strip()

    return df


def one_hot_encode_skills_and_activities(df):
    """
    One-hot encodes the skills and work activities columns for use in machine learning models.

    Args:
        df (pd.DataFrame): DataFrame containing 'Skills' and 'Work Activities' columns.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded skills and work activities.
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # One-hot encode the 'Skills' and 'Work Activities' columns
    skills_encoded = encoder.fit_transform(df[['Skills']])
    work_activities_encoded = encoder.fit_transform(df[['Work Activities']])

    # Convert the encoded arrays back into DataFrames with proper column names
    skills_df = pd.DataFrame(skills_encoded, columns=encoder.get_feature_names_out(['Skills']))
    work_activities_df = pd.DataFrame(work_activities_encoded,
                                      columns=encoder.get_feature_names_out(['Work Activities']))

    # Reset index to align the DataFrames for merging
    df.reset_index(drop=True, inplace=True)
    skills_df.reset_index(drop=True, inplace=True)
    work_activities_df.reset_index(drop=True, inplace=True)

    # Merge the one-hot encoded data back into the original DataFrame
    df_encoded = pd.concat([df, skills_df, work_activities_df], axis=1)

    # Drop the original 'Skills' and 'Work Activities' columns
    df_encoded.drop(columns=['Skills', 'Work Activities'], inplace=True)

    return df_encoded


def normalize_columns(df, columns_to_normalize):
    """
    Normalizes the specified columns in the DataFrame using StandardScaler.

    Args:
        df (pd.DataFrame): DataFrame containing columns to normalize.
        columns_to_normalize (list): List of column names to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df


def preprocess_existing_industry_data():
    """
    Preprocesses the existing industry data by cleaning, one-hot encoding, and normalizing.

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for analysis.
    """
    # Load raw data collected in data_collection.py
    from data_collection import collect_existing_industry_data
    raw_data = collect_existing_industry_data()

    # Step 1: Clean the data
    cleaned_data = clean_data(raw_data)

    # Step 2: One-hot encode categorical variables (Skills and Work Activities)
    encoded_data = one_hot_encode_skills_and_activities(cleaned_data)

    # Step 3: Normalize numeric columns (add columns that need normalization)
    # Example: Assume 'SOC Code' and other numeric features (customize this step as needed)
    numeric_columns = ['SOC Code']  # Customize based on your dataset
    normalized_data = normalize_columns(encoded_data, numeric_columns)

    # Save the preprocessed data
    preprocessed_file_path = os.path.join(PROCESSED_DIR, "preprocessed_existing_industries.csv")
    normalized_data.to_csv(preprocessed_file_path, index=False)

    print(f"Preprocessed data saved to {preprocessed_file_path}")
    return normalized_data


def preprocess_target_industry_data(target_industry_soc_code):
    """
    Preprocesses the target industry data by cleaning, one-hot encoding, and normalizing.

    Args:
        target_industry_soc_code (str): SOC code of the target industry.

    Returns:
        pd.DataFrame: Preprocessed DataFrame for the target industry.
    """
    # Load raw target industry data collected in data_collection.py
    from data_collection import collect_target_industry_data
    raw_target_data = collect_target_industry_data(target_industry_soc_code)

    # Convert to DataFrame
    raw_target_df = pd.DataFrame([raw_target_data])

    # Step 1: Clean the data
    cleaned_target_data = clean_data(raw_target_df)

    # Step 2: One-hot encode categorical variables (Skills and Work Activities)
    encoded_target_data = one_hot_encode_skills_and_activities(cleaned_target_data)

    # Step 3: Normalize numeric columns (if applicable)
    # Example: Assume 'SOC Code' is a numeric feature to normalize
    numeric_columns = ['SOC Code']  # Customize based on your dataset
    normalized_target_data = normalize_columns(encoded_target_data, numeric_columns)

    # Save the preprocessed target data
    preprocessed_target_file_path = os.path.join(PROCESSED_DIR, "preprocessed_target_industry.csv")
    normalized_target_data.to_csv(preprocessed_target_file_path, index=False)

    print(f"Preprocessed target industry data saved to {preprocessed_target_file_path}")
    return normalized_target_data


if __name__ == "__main__":
    # Example usage:

    # Preprocess existing industry data
    preprocessed_existing_data = preprocess_existing_industry_data()
    print("Preprocessed Existing Industry Data:")
    print(preprocessed_existing_data.head())

    # Preprocess target industry data (example SOC code: '15-1132.00')
    target_industry_soc_code = '15-1132.00'
    preprocessed_target_data = preprocess_target_industry_data(target_industry_soc_code)
    print("\nPreprocessed Target Industry Data:")
    print(preprocessed_target_data.head())
