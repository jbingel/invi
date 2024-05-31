import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, pdist
import scipy 

from visualization import create_visualizations

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


def write_report_file(output_dir: str, question: str, response_type: str, condensed: bool, condensed_responses: pd.Series, importance_weighting: bool, df_results: pd.DataFrame):
    report_file = os.path.join(output_dir, f"{response_type}_analysis.md")
    print(f"Writing report to {report_file}\n")

    with open(report_file, "w") as f:
        f.write(f"# {response_type.capitalize()} analysis\n\n")
        f.write(f"Question: {question}\n")
        f.write(f"Using condensed answers for analysis: {'Yes' if condensed else 'No'}\n")
        f.write(f"Using importance weighting: {'Yes' if importance_weighting else 'No'}\n")
        f.write("\n")
        # Write value counts of condensed_response column
        f.write(f"## Overview of responses\n")
        f.write(f"Value counts of condensed responses (first 30):\n")
        f.write(condensed_responses.value_counts()[:30].to_markdown())
        f.write("\n\n")
        f.write(f"## Metrics \n\n {df_results.to_markdown()}")

    return report_file


def analyze(
        augmented_data_path: str, 
        question: str, 
        response_type: str, 
        condensed: bool,
        importance_weighting: bool,
        visualize: bool, 
        output_dir: str,
        do_write_report: bool = True,
    ):

    # Read CSV with augmented data
    df = pd.read_csv(augmented_data_path)

    # Remove irrelevant responses
    if "relevant" in df.columns:
        df = df[df.relevant == True]

    # Prepare the embeddings
    response_column = "response" if not condensed else "condensed_response_max5" 
    embedding_column = "response_embedding" if not condensed else "condensed_response_embedding_max5"

    if importance_weighting:
        print(f"Dataframe has {len(df)} rows before weighting")
        df = df.loc[df.index.repeat(df['weight'])]
        print(f"Dataframe has {len(df)} rows after weighting")

    embeddings = np.vstack(df[embedding_column].apply(lambda x: np.array(eval(x))).tolist())
    

    ## ANALYSIS ##
    results = []

    # Metric: average distance from mean
    # Convert the list of embeddings into a numpy array
    mean = np.mean(embeddings, axis=0)
    # Calculate the cosine distance of each embedding to the mean embedding
    cosine_distances = np.array([cosine(embedding, mean) for embedding in embeddings])
    avg_cosine_distance = np.mean(cosine_distances)
    results.append({
        "metric": "Avg. distance from mean",
        "value": avg_cosine_distance,
    })

    # Metric: variance
    # Calculate the variance of the cosine distances
    variance = np.var(cosine_distances)
    results.append({
        "metric": "Variance of distance from mean",
        "value": variance,
    })

    # Metric: mean pairwise distance
    # Calculate mean pairwise distance
    pairwise_cosine_distances = pdist(embeddings, metric='cosine')
    results.append({
        "metric": "Mean pairwise cosine distance",
        "value": np.mean(pairwise_cosine_distances),
    })


    # Metric: mean pairwise distance
    # Calculate mean pairwise distance
    pairwise_euclidean_distances = pdist(embeddings, metric='euclidean')
    results.append({
        "metric": "Mean pairwise Euclidean distance",
        "value": np.mean(pairwise_euclidean_distances),
    })

    # Metric: Gini coefficient
    # Calculate the Gini coefficient
    gini = rowwise_gini(normalize(embeddings)) 
    results.append({
        "metric": "Gini",
        "value": gini,
    })

    # Metric: entropy
    # Calculate the entropy
    entropy = scipy.special.entr(normalize(embeddings)).sum(axis=1).mean()
    results.append({
        "metric": "Entropy",
        "value": entropy,
    })

    # Metric: Determinant of vector/vector distance matrix
    M = embeddings.dot(embeddings.T)  # distance matrix
    determinant = np.linalg.det(M)
    results.append({
        "metric": "Determinant",
        "value": determinant,
    })

    if visualize:
        text_labels = df[response_column].dropna().values
        create_visualizations(
            df, embeddings, output_dir, 
            response_type=response_type, 
            text_labels=text_labels, 
            question=question
        )

    df_results = pd.DataFrame(results)

    print(df_results.to_markdown())

    # Write results to markdown    
    if do_write_report:
        write_report_file(output_dir, question, response_type, condensed, df['condensed_response'], importance_weighting, df_results)
            

    return results, pairwise_cosine_distances

