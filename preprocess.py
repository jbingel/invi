import pandas as pd
from tqdm import tqdm

from text_processing import get_embedding, condense_response

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
                condensed = condense_response(cause, cause_question)
                combined_causes.append({
                    "response": cause,
                    "condensed_response": condensed,
                    "original_index": idx,
                    "response_embedding": get_embedding(cause),
                    "condensed_response_embedding": get_embedding(condensed),
                    "weight": weight
                })
        for col in solution_cols:
            solution = row[col]
            if pd.notnull(solution):
                condensed = condense_response(solution, solution_question)
                combined_solutions.append({
                    "response": solution,
                    "condensed_response": condensed,
                    "original_index": idx,
                    "response_embedding": get_embedding(solution),
                    "condensed_response_embedding": get_embedding(condensed),
                    "weight": weight
                })


    print(f"Saving augmented data to {output_dir}")
    pd.DataFrame(combined_causes).to_csv(f"{output_dir}/causes_augmented.csv")
    pd.DataFrame(combined_solutions).to_csv(f"{output_dir}/solutions_augmented.csv")
    print("Done")