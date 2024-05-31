import json
import pandas as pd
from tqdm import tqdm

from text_processing import get_embedding, augment_response, seperate_complex_responses

# Define a function to create files based on an Excel file and questions
def augment_csv(
        input_file: str, 
        cause_question: str, 
        solution_question: str, 
        weighting_question: str,
        output_dir: str,
    ):
    
    print("Reading excel file")
    # Read an Excel file and select a specific sheet
    df = pd.read_excel(input_file, sheet_name='Complete')
    
    # df = df.head(10)

    print(f"DataFrame contains {len(df)} entries.")
    
    # Mapping from textual answers to numerical values
    mapping = {
        "Ved ikke": 0,
        "I meget lav grad": 1,
        "I lav grad": 2,
        "Hverken eller ": 3,
        "I høj grad ": 4,
        "I meget høj grad": 5
    }

    df.columns = df.columns.str.strip()

    df = df.dropna(subset=[cause_question, solution_question, weighting_question])

    # Apply mapping to a specific question in the DataFrame to filter data
    df[weighting_question] = df[weighting_question].map(mapping)

    print(f"The filtered DataFrame contains {len(df)} entries.")

    print("Finding relevant columns")
    cause_cols = [col for col in df.columns if col.startswith(cause_question)]
    solution_cols = [col for col in df.columns if col.startswith(solution_question)]

    combined_causes = []
    combined_solutions = []

    print("Processing responses. Condensing and embedding...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        for col in cause_cols:
            cause = row[col]
            weight = row[weighting_question]

            if pd.notnull(cause):
                for output in augment_response(cause, cause_question):
                    condensed, condensed_max4, condensed_max5, relevant = output
                    
                    combined_causes.append({
                        "weight": weight,
                        "relevant": relevant,
                        "condensed_response": condensed,
                        "condensed_response_max4": condensed_max4,
                        "condensed_response_max5": condensed_max5,
                        "response": cause,
                        "original_index": idx,
                        "response_embedding": get_embedding(cause),
                        "condensed_response_embedding": get_embedding(condensed),
                        "condensed_response_embedding_max4": get_embedding(condensed_max4),
                        "condensed_response_embedding_max5": get_embedding(condensed_max5),
                    })
        for col in solution_cols:
            solution = row[col]
            if pd.notnull(solution):
                for output in augment_response(solution, solution_question):
                    condensed, condensed_max5, condensed_max4, relevant = output

                    combined_solutions.append({
                        "weight": weight,
                        "relevant": relevant,
                        "condensed_response": condensed,
                        "condensed_response_max4": condensed_max4,
                        "condensed_response_max5": condensed_max5,
                        "response": solution,
                        "original_index": idx,
                        "response_embedding": get_embedding(solution),
                        "condensed_response_embedding": get_embedding(condensed),
                        "condensed_response_embedding_max4": get_embedding(condensed_max4),
                        "condensed_response_embedding_max5": get_embedding(condensed_max5),
                    })


    print(f"Saving augmented data to {output_dir}")
    pd.DataFrame(combined_causes).to_csv(f"{output_dir}/causes_augmented.csv")
    pd.DataFrame(combined_solutions).to_csv(f"{output_dir}/solutions_augmented.csv")
    print("Done")