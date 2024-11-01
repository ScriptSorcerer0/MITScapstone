import pandas as pd
from sklearn.metrics import jaccard_score

def get_common_dwa_ids(threshold=0.5):
    tasks_to_dwa = pd.read_excel("Tasks to DWAs.xlsx")
    total_naics = tasks_to_dwa['SOC Code'].nunique()
    common_dwa_counts = tasks_to_dwa.groupby('DWA ID')['SOC Code'].nunique()
    return set(common_dwa_counts[common_dwa_counts > total_naics * threshold].index)

def calculate_jaccard_dwa_similarity(naics_code1, naics_code2, common_dwa_ids):
    naics_to_soc = pd.read_excel("NAICS_to_SOC.xlsx")
    soc_codes1 = naics_to_soc[naics_to_soc['NAICS'] == naics_code1]['OCC_CODES'].str.split(', ').explode().drop_duplicates().str.split('.').str[0]
    soc_codes2 = naics_to_soc[naics_to_soc['NAICS'] == naics_code2]['OCC_CODES'].str.split(', ').explode().drop_duplicates().str.split('.').str[0]

    tasks_to_dwa = pd.read_excel("Tasks to DWAs.xlsx")
    soc_column = 'SOC Code' if 'SOC Code' in tasks_to_dwa.columns else 'SOC_Code'
    tasks_to_dwa[soc_column] = tasks_to_dwa[soc_column].astype(str).str.split('.').str[0]

    dwa_naics1 = set(tasks_to_dwa[tasks_to_dwa[soc_column].isin(soc_codes1)]['DWA ID']) - common_dwa_ids
    dwa_naics2 = set(tasks_to_dwa[tasks_to_dwa[soc_column].isin(soc_codes2)]['DWA ID']) - common_dwa_ids
    all_dwas = list(dwa_naics1 | dwa_naics2)
    vector1 = [1 if dwa in dwa_naics1 else 0 for dwa in all_dwas]
    vector2 = [1 if dwa in dwa_naics2 else 0 for dwa in all_dwas]
    print("Industry Detailed Work Activities Similarity: ", jaccard_score(vector1, vector2).round(2))
    return jaccard_score(vector1, vector2)
