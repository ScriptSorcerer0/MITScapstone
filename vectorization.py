import numpy as np
from sklearn.preprocessing import OneHotEncoder
from data_collection import collect_existing_industry_data, collect_target_industry_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def create_feature_vectors(data):
    encoder = OneHotEncoder(sparse=True)
    encoded_data = encoder.fit_transform(data)
    return encoded_data





def create_tfidf_vectors(text_data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_data)
    svd = TruncatedSVD(n_components=100)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    return reduced_matrix

def vectorize_existing_industries():
    industry_data = collect_existing_industry_data().to_numpy()
    skills_col = industry_data[:, 1]
    assets_col = industry_data[:, 2]
    industry_vectors = create_feature_vectors(industry_data)
    skill_vectors = create_tfidf_vectors(skills_col)
    asset_vectors = create_tfidf_vectors(assets_col)
    return np.hstack([industry_vectors.toarray(), skill_vectors, asset_vectors])

def vectorize_target_industry():
    target_data = collect_target_industry_data('15-1132.00')
    target_vectors = create_feature_vectors(np.array(list(target_data.values())))
    return target_vectors.toarray()
