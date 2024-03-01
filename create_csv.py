import pandas as pd
from openai import OpenAI
from constants import OPENAI_API_KEY
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialiserer OpenAI klienten med en API-nøgle
client = OpenAI(api_key=OPENAI_API_KEY)

# Definerer en funktion til at skabe filer baseret på en Excel-fil og spørgsmål
def create_files(xlsx_path, cause_question, solution_question, filter_question, filter_value):
    print("Reading excel file")
    # Læser en Excel-fil og vælger et specifikt ark
    df = pd.read_excel(xlsx_path, sheet_name='Complete')
    
    # Printer antallet af besvarelser i den indlæste dataframe
    print(f"Dataframen indeholder {len(df)} besvarelser.")
    # Definerer en mapping fra tekstuelle svar til numeriske værdier
    mapping = {
        "Ved ikke": 0,
        "I meget lav grad": 1,
        "I lav grad": 2,
        "Hverken eller": 3,
        "I høj grad": 4,
        "I meget høj grad": 5
    }

    # Anvender mapping på et specifikt spørgsmål i dataframen for at filtrere data
    df[filter_question] = df[filter_question].map(mapping)

    # Filtrer dataframen baseret på en specifik værdi
    df = df[df[filter_question] <= filter_value]

    # Printer antallet af besvarelser efter filtrering
    print(f"Den filtrerede dataframe indeholder {len(df)} besvarelser.")

    # Definerer en funktion til at hente embeddings for en given tekst
    def get_embedding(text, model="text-embedding-3-large"):
        if isinstance(text, str):
            text = text.replace("\n", " ")
            return client.embeddings.create(input = [text], model=model, dimensions=256).data[0].embedding
        
    # Definerer en funktion til at beregne cosine similarity mellem to vektorer
    def calculate_cosine_similarity(answer, question_embedding):
        a = np.array([answer])
        b = np.array([question_embedding])
        cosine_similarity_result = cosine_similarity(a, b)
        return cosine_similarity_result

    # Skaber embeddings for de angivne spørgsmål
    print("Creating embeddings for questions")
    cause_question_embedding = get_embedding(cause_question)
    solution_question_embedding = get_embedding(solution_question)

    # Finder relevante kolonner baseret på de angivne spørgsmål
    print("Finding relevant columns")
    causes = [col for col in df.columns if col.startswith(cause_question)]
    solutions = [col for col in df.columns if col.startswith(solution_question)]

    # Initialiserer lister for at gemme kombinerede tekster fra hver kategori
    combined_causes = []
    combined_solutions = []
    
    # Itererer over hver række i df for at kombinere tekster fra årsags- og løsningskolonner
    print("Get all text from those columns")
    for cause in causes:
        texts = df[cause].dropna().values
        combined_causes.extend(texts)

    for solution in solutions:
        texts = df[solution].dropna().values
        combined_solutions.extend(texts)

    # Skaber nye dataframes med de kombinerede tekster
    print("Creating new dataframes with the texts")
    cause_df = pd.DataFrame({
        'question': combined_causes,
    })

    solution_df = pd.DataFrame({
        'question': combined_solutions,
    })

    # Beregner nye embeddings for årsager og afstand til det oprindelige spørgsmål
    print("Calculating new embeddings for cause")
    cause_df['question_embedding'] = cause_df['question'].apply(lambda x: get_embedding(x, model='text-embedding-3-large'))
    print("Calculating distance to question")
    cause_df['distance'] = cause_df['question_embedding'].apply(lambda x: calculate_cosine_similarity(x, cause_question_embedding))
    cause_df.to_csv('cause_output.csv', index=False)

    # Gør det samme for løsninger
    print("Calculating new embeddings for solution")
    solution_df['question_embedding'] = solution_df['question'].apply(lambda x: get_embedding(x, model='text-embedding-3-large'))
    print("Calculating distance to question")
    solution_df['distance'] = solution_df['question_embedding'].apply(lambda x: calculate_cosine_similarity(x, solution_question_embedding))
    solution_df.to_csv('solution_output.csv', index=False)

    return 'cause_output.csv', 'solution_output.csv'