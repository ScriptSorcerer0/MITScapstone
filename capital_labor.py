import pandas as pd

def standardize_naics_code(naics_code):

    naics_code = str(naics_code)
    return naics_code.ljust(6, '0')

def calculate_capital_labor_similarity(naics_code1, naics_code2):
    naics_code1 = standardize_naics_code(naics_code1)
    naics_code2 = standardize_naics_code(naics_code2)

    data = pd.read_excel("labor.xlsx")

    data['NAICS'] = data['NAICS'].apply(standardize_naics_code)

    output_per_worker_2021 = data[(data['Measure'] == 'Output per worker') & (data['Units'] == 'Index (2017=100)') & (data['2021'].notna())]
    hourly_compensation_2021 = data[(data['Measure'] == 'Hourly compensation') & (data['Units'] == 'Index (2017=100)') & (data['2021'].notna())]



    merged_data_2021 = pd.merge(
        output_per_worker_2021[['NAICS', 'Industry', '2021']],
        hourly_compensation_2021[['NAICS', 'Industry', '2021']],
        on=['NAICS', 'Industry'],
        suffixes=('_output', '_compensation')
    )

    merged_data_2021['Capital_Labor_2021'] = merged_data_2021['2021_output'] / merged_data_2021['2021_compensation']

    min_val = merged_data_2021['Capital_Labor_2021'].min()
    max_val = merged_data_2021['Capital_Labor_2021'].max()
    merged_data_2021['Capital_Labor_2021_normalized'] = (merged_data_2021['Capital_Labor_2021'] - min_val) / (max_val - min_val)

    capital_labor_1 = merged_data_2021[merged_data_2021['NAICS'] == naics_code1]['Capital_Labor_2021_normalized']
    capital_labor_2 = merged_data_2021[merged_data_2021['NAICS'] == naics_code2]['Capital_Labor_2021_normalized']



    if capital_labor_1.empty or capital_labor_2.empty:
        return 0.0

    similarity_score = 1 - abs(capital_labor_1.values[0] - capital_labor_2.values[0])

    print(similarity_score.round(2))
    return max(0.0, min(1.0, similarity_score))