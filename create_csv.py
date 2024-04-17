import pandas as pd
from openai import OpenAI
from constants import OPENAI_API_KEY
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ast
import numpy as np

# Initialize OpenAI client with an API key
client = OpenAI(api_key=OPENAI_API_KEY)

def add_cluster_labels(embeddings):
    # Convert embeddings from strings to numpy arrays if necessary
    # embeddings = embeddings.apply(ast.literal_eval)
    embeddings_array = np.stack(embeddings.values)
    
    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_array)
    
    # Determine the optimal number of clusters
    silhouette_scores = []
    range_of_clusters = range(2, 20)
    best_num_clusters = 2
    best_silhouette_score = -1
    for k in range_of_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
        silhouette_avg = silhouette_score(embeddings_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = k
            
    # Cluster with the optimal number of clusters
    kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)
    
    return cluster_labels

# Define a function to create files based on an Excel file and questions
def create_files(xlsx_path, cause_question, solution_question, filter_question, filter_value):
    print("Reading excel file")
    # Read an Excel file and select a specific sheet
    df = pd.read_excel(xlsx_path, sheet_name='Complete')
    
    print(f"DataFrame contains {len(df)} entries.")
    
    # Mapping from textual answers to numerical values
    if filter_value != -1:
        mapping = {
            "Ved ikke": 0,
            "I meget lav grad": 1,
            "I lav grad": 2,
            "Hverken eller ": 3,
            "I høj grad ": 4,
            "I meget høj grad": 5
        }

        # Apply mapping to a specific question in the DataFrame to filter data
        df[filter_question] = df[filter_question].map(mapping)
        df = df[df[filter_question] >= filter_value]
    
    print(f"The filtered DataFrame contains {len(df)} entries.")

    def get_embedding(text, model="text-embedding-3-large"):
        if isinstance(text, str):
            text = text.replace("\n", " ")
            return client.embeddings.create(input = [text], model=model, dimensions=256).data[0].embedding
        
    def calculate_cosine_similarity(answer, question_embedding):
        a = np.array([answer])
        b = np.array([question_embedding])
        cosine_similarity_result = cosine_similarity(a, b)
        return cosine_similarity_result

    print("Creating embeddings for questions")
    cause_question_embedding = get_embedding(cause_question)
    solution_question_embedding = get_embedding(solution_question)

    print("Finding relevant columns")
    causes = [col for col in df.columns if col.startswith(cause_question)]
    solutions = [col for col in df.columns if col.startswith(solution_question)]

    combined_causes = []
    combined_solutions = []
    
    print("Get all text from those columns")
    for idx, row in df.iterrows():
        for cause in causes:
            if pd.notnull(row[cause]):
                combined_causes.append((row[cause], idx))
        for solution in solutions:
            if pd.notnull(row[solution]):
                combined_solutions.append((row[solution], idx))

    print("Creating new dataframes with the texts")
    cause_df = pd.DataFrame(combined_causes, columns=['question', 'original_index'])
    solution_df = pd.DataFrame(combined_solutions, columns=['question', 'original_index'])

    print("Calculating new embeddings for cause")
    cause_df['question_embedding'] = cause_df['question'].apply(lambda x: get_embedding(x, model='text-embedding-3-large'))

    print("Calculating distance to question")
    cause_df['distance'] = cause_df['question_embedding'].apply(lambda x: calculate_cosine_similarity(x, cause_question_embedding))
    cause_df.to_csv('cause_output.csv', index=False)

    print("Calculating new embeddings for solution")
    solution_df['question_embedding'] = solution_df['question'].apply(lambda x: get_embedding(x, model='text-embedding-3-large'))

    cause_df['cluster_label'] = add_cluster_labels(cause_df['question_embedding'])
    solution_df['cluster_label'] = add_cluster_labels(solution_df['question_embedding'])

    print("Calculating distance to question")
    solution_df['distance'] = solution_df['question_embedding'].apply(lambda x: calculate_cosine_similarity(x, solution_question_embedding))
    solution_df.to_csv('solution_output.csv', index=False)

    return 'cause_output.csv', 'solution_output.csv'
