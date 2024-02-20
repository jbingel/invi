import pandas as pd
from openai import OpenAI
from constants import OPENAI_API_KEY
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

client = OpenAI(api_key=OPENAI_API_KEY)

# Read the excel file

def create_files(xlsx_path, cause_question, solution_question):
    print("Reading excel file")
    df = pd.read_excel(xlsx_path, sheet_name='Complete')

    def get_embedding(text, model="text-embedding-3-large"):
        if isinstance(text, str):
            text = text.replace("\n", " ")
            return client.embeddings.create(input = [text], model=model, dimensions=256).data[0].embedding
        
    def calculate_cosine_similarity(answer, question_embedding):
        a = np.array([answer])
        b = np.array([question_embedding])
        cosine_similarity_result = cosine_similarity(a, b)
        return cosine_similarity_result

    # Embeddings
    print("Creating embeddings for questions")
    cause_question_embedding = get_embedding(cause_question)
    solution_question_embedding = get_embedding(solution_question)

    # Find all relevant columns
    print("Finding relevant columns")
    causes = [col for col in df.columns if col.startswith(cause_question)]
    solutions = [col for col in df.columns if col.startswith(solution_question)]

    # Initialize lists to store combined texts from each category
    combined_causes = []
    combined_solutions = []

    # Iterate over each row in df to combine texts from cause and solution columns
    print("Get all text from those columns")
    for cause in causes:
        texts = df[cause].dropna().values
        combined_causes.extend(texts)

    for solution in solutions:
        texts = df[solution].dropna().values
        combined_solutions.extend(texts)


    # Create a new DataFrame with combined texts
    print("Creating new dataframes with the texts")
    cause_df = pd.DataFrame({
        'question': combined_causes,
    })

    solution_df = pd.DataFrame({
        'question': combined_solutions,
    })

    print("Calculating new embeddings for cause")
    cause_df['question_embedding'] = cause_df['question'].apply(lambda x: get_embedding(x, model='text-embedding-3-large'))
    print("Calculating distance to question")
    cause_df['distance'] = cause_df['question_embedding'].apply(lambda x: calculate_cosine_similarity(x, cause_question_embedding))
    cause_df.to_csv('cause_output.csv', index=False)

    print("Calculating new embeddings for solution")
    solution_df['question_embedding'] = solution_df['question'].apply(lambda x: get_embedding(x, model='text-embedding-3-large'))
    print("Calculating distance to question")
    solution_df['distance'] = solution_df['question_embedding'].apply(lambda x: calculate_cosine_similarity(x, solution_question_embedding))
    solution_df.to_csv('solution_output.csv', index=False)

    return 'cause_output.csv', 'solution_output.csv'