import pandas as pd
import numpy as np
def calculate_abilities_difference(naics_code1, naics_code2):
    naics_to_soc = pd.read_excel("NAICS_to_SOC.xlsx")
    soc_codes1 = naics_to_soc[naics_to_soc['NAICS'] == naics_code1]['OCC_CODES'].str.split(', ').explode().drop_duplicates().str.split('.').str[0]
    soc_codes2 = naics_to_soc[naics_to_soc['NAICS'] == naics_code2]['OCC_CODES'].str.split(', ').explode().drop_duplicates().str.split('.').str[0]

    abilities_data = pd.read_excel("abilities.xlsx")
    soc_column = 'SOC Code' if 'SOC Code' in abilities_data.columns else 'SOC_Code'
    abilities_data[soc_column] = abilities_data[soc_column].astype(str).str.split('.').str[0]

    abilities1 = abilities_data[abilities_data[soc_column].isin(soc_codes1)]
    abilities2 = abilities_data[abilities_data[soc_column].isin(soc_codes2)]
    avg_abilities1 = abilities1.groupby(['Element ID', 'Scale ID'])['Data Value'].mean()
    avg_abilities2 = abilities2.groupby(['Element ID', 'Scale ID'])['Data Value'].mean()
    ability_differences = (avg_abilities1 - avg_abilities2).abs()

    return ability_differences.mean()

def calculate_skills_difference(naics_code1, naics_code2):
    naics_to_soc = pd.read_excel("NAICS_to_SOC.xlsx")
    soc_codes1 = naics_to_soc[naics_to_soc['NAICS'] == naics_code1]['OCC_CODES'].str.split(', ').explode().drop_duplicates().str.split('.').str[0]
    soc_codes2 = naics_to_soc[naics_to_soc['NAICS'] == naics_code2]['OCC_CODES'].str.split(', ').explode().drop_duplicates().str.split('.').str[0]

    skills_data = pd.read_excel("Skills.xlsx")
    soc_column = 'SOC Code' if 'SOC Code' in skills_data.columns else 'SOC_Code'
    skills_data[soc_column] = skills_data[soc_column].astype(str).str.split('.').str[0]

    skills1 = skills_data[skills_data[soc_column].isin(soc_codes1)]
    skills2 = skills_data[skills_data[soc_column].isin(soc_codes2)]
    avg_skills1 = skills1.groupby(['Element ID', 'Scale ID'])['Data Value'].mean()
    avg_skills2 = skills2.groupby(['Element ID', 'Scale ID'])['Data Value'].mean()
    skill_differences = (avg_skills1 - avg_skills2).abs()

    return skill_differences.mean()

def calculate_knowledge_difference(naics_code1, naics_code2):
    naics_to_soc = pd.read_excel("NAICS_to_SOC.xlsx")
    soc_codes1 = naics_to_soc[naics_to_soc['NAICS'] == naics_code1]['OCC_CODES'].str.split(', ').explode().drop_duplicates().str.split('.').str[0]
    soc_codes2 = naics_to_soc[naics_to_soc['NAICS'] == naics_code2]['OCC_CODES'].str.split(', ').explode().drop_duplicates().str.split('.').str[0]

    knowledge_data = pd.read_excel("Knowledge.xlsx")
    soc_column = 'SOC Code' if 'SOC Code' in knowledge_data.columns else 'SOC_Code'
    knowledge_data[soc_column] = knowledge_data[soc_column].astype(str).str.split('.').str[0]

    knowledge1 = knowledge_data[knowledge_data[soc_column].isin(soc_codes1)]
    knowledge2 = knowledge_data[knowledge_data[soc_column].isin(soc_codes2)]
    avg_knowledge1 = knowledge1.groupby(['Element ID', 'Scale ID'])['Data Value'].mean()
    avg_knowledge2 = knowledge2.groupby(['Element ID', 'Scale ID'])['Data Value'].mean()
    knowledge_differences = (avg_knowledge1 - avg_knowledge2).abs()

    return knowledge_differences.mean()

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
def calculate_worker_difference(naics_code1, naics_code2):
    naics_to_soc = pd.read_excel("NAICS_to_SOC.xlsx")
    soc_codes1 = naics_to_soc[naics_to_soc['NAICS'] == naics_code1]['OCC_CODES'].str.split(', ').explode().drop_duplicates().str.split('.').str[0]
    soc_codes2 = naics_to_soc[naics_to_soc['NAICS'] == naics_code2]['OCC_CODES'].str.split(', ').explode().drop_duplicates().str.split('.').str[0]

    ete_data = pd.read_excel("ETE.xlsx")
    soc_column = 'SOC Code' if 'SOC Code' in ete_data.columns else 'SOC_Code'
    ete_data[soc_column] = ete_data[soc_column].astype(str).str.split('.').str[0]

    ete1 = ete_data[ete_data[soc_column].isin(soc_codes1)]
    ete2 = ete_data[ete_data[soc_column].isin(soc_codes2)]
    avg_ete1 = ete1.groupby(['Scale ID', 'Category'])['Data Value'].mean()
    avg_ete2 = ete2.groupby(['Scale ID', 'Category'])['Data Value'].mean()
    ete_differences = (avg_ete1 - avg_ete2).abs()
    print(ete_differences.mean())
    return ete_differences.mean()



# Function to calculate the final worker-related similarity score
def calculate_worker_similarity(naics_code1, naics_code2):
    abilities_sim = 1-calculate_abilities_difference(naics_code1, naics_code2)
    skills_sim = 1-calculate_skills_difference(naics_code1, naics_code2)
    knowledge_sim = 1-calculate_knowledge_difference(naics_code1, naics_code2)
    worker_sim = calculate_worker_difference(naics_code1, naics_code2)
    print('Industry Worker Ability Similarity: ', abilities_sim.round(2))
    print('Industry Worker Skills Similarity: ', skills_sim.round(2))
    print('Industry Worker Knowledge Similarity: ', knowledge_sim.round(2)),
    print('Industry Worker Education, Experience, and Training Similarity: ', worker_sim.round(2))
    # Average the differences for an overall worker similarity score
    return (abilities_sim + skills_sim + knowledge_sim + worker_sim) / 4

