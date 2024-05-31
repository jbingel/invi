import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, pdist
import scipy 

from visualization import create_visualizations
from text_processing import get_embedding

from tqdm import tqdm
tqdm.pandas()

def normalize(X):
    # normalize matrix rowwise to [0, 1]
    X = (X-np.min(X))/(np.max(X)-np.min(X))
    X = X / X.sum(keepdims=True)
    return X

def rowwise_gini(X):
    def gini(x):
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(x, x)).mean()
        # Relative mean absolute difference
        rmad = mad/np.mean(x)
        # Gini coefficient
        g = 0.5 * rmad
        return g
    
    g = []
    for col in X.T:
        g.append(gini(col))
    return np.mean(g)

def analyze(
        augmented_data_path: str, 
        question: str, 
        response_type: str, 
        condensed: bool,
        importance_weighting: bool,
        visualize: bool, 
        output_dir: str
    ):

    df = pd.read_csv(augmented_data_path)
    if "relevant" in df.columns:
        df = df[df.relevant == True]

    report_file = os.path.join(output_dir, f"{response_type}_analysis.md")
    print(f"Writing report to {report_file}")

    with open(report_file, "w") as f:
        f.write(f"# {response_type.capitalize()} analysis\n\n")
        f.write(f"Question: {question}\n")
        f.write(f"Using condensed answers for analysis: {'Yes' if condensed else 'No'}\n")
        f.write("\n")
        # Write value counts of condensed_response column
        f.write(f"## Overview of responses\n")
        f.write(f"Value counts of condensed responses:\n")
        f.write(df["condensed_response"].value_counts().to_markdown())
        f.write("\n")

        f.write

    # Prepare the embeddings
    response_column = "response" if not condensed else "condensed_response_max5" 
    embedding_column = "response_embedding" if not condensed else "condensed_response_embedding_max5"

    print(response_column)

    if importance_weighting:
        print(f"Dataframe has {len(df)} rows before weighting")
        df = df.loc[df.index.repeat(df['weight'])]
        print(f"Dataframe has {len(df)} rows after weighting")

    print(f"Getting embeddings for reponse column {response_column}")
    # df[embedding_column] = df[response_column].apply(get_embedding)
    # embeddings = np.vstack(df[embedding_column])s
    embeddings = np.vstack(df[embedding_column].apply(lambda x: np.array(eval(x))).tolist())
    

    # Text labels for each point from "s_8" column
    text_labels = df[response_column].dropna().values
    print("DATA EVALUATED")

    # Convert the list of embeddings into a numpy array
    mean = np.mean(embeddings, axis=0)

    # Calculate the cosine distance of each embedding to the mean embedding
    cosine_distances = np.array([cosine(embedding, mean) for embedding in embeddings])
    avg_cosine_distance = np.mean(cosine_distances)
    variance = np.var(cosine_distances)

    # Calculate mean pairwise distance
    pairwise_distances = pdist(embeddings, metric='cosine')
    # Compute the mean of these distances
    mean_pairwise_distance = np.mean(pairwise_distances)


    # Calculate the Gini coefficient
    gini = rowwise_gini(normalize(embeddings)) 

    # Calculate the entropy
    entropy = scipy.special.entr(
        normalize(embeddings)
    ).sum(axis=1).mean()

    # Determinant of vector/vector distance matrix
    M = embeddings.dot(embeddings.T)
    determinant = np.linalg.det(M)

    if visualize:
        
        create_visualizations(
            df, embeddings, output_dir, 
            response_type=response_type, 
            text_labels=text_labels, 
            question=question
        )

    results = pd.DataFrame([
        {
            "metric": "Avg. distance from mean",
            "value": avg_cosine_distance
        },
        {
            "metric": "Variance",
            "value": variance
        },
        {
            "metric": "Mean pairwise distance",
            "value": mean_pairwise_distance
        },
        {
            "metric": "Entropy",
            "value": entropy
        },
        {
            "metric": "Gini",
            "value": gini
        },
        {
            "metric": "Determinant",
            "value": determinant
        }
    ])

    print(results.to_markdown())
    
    with open(os.path.join(output_dir, f"{response_type}_analysis.md"), "a") as f:
        f.write("\n\n")
        f.write("## Metrics \n\n")
        f.write(results.to_markdown())

