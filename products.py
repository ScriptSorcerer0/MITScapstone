import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_napcs_similarity(naics_code1, naics_code2):
    data = pd.read_excel("NAICS_to_NAPCS_Cleaned.xlsx")

    data['NAPCS Codes'] = data['NAPCS Codes'].apply(lambda x: ' '.join(x.split(', ')))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['NAPCS Codes'])

    index1 = data.index[data['NAICS Code'] == naics_code1].tolist()[0]
    index2 = data.index[data['NAICS Code'] == naics_code2].tolist()[0]

    similarity_score = cosine_similarity(tfidf_matrix[index1], tfidf_matrix[index2])[0][0]
    print("Industry Product Similarity: ", similarity_score.round(2))
    return similarity_score
