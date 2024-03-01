import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
from openai import OpenAI
from constants import OPENAI_API_KEY
from scipy.spatial.distance import cdist, cosine

client = OpenAI(api_key=OPENAI_API_KEY)


def create_visualizations(data: str, question: str, visualization_type: str):
    data = pd.read_csv(data)

    # Prepare the embeddings
    embeddings = data['question_embedding'].dropna()
    embeddings = embeddings.apply(ast.literal_eval)
    # Text labels for each point from "s_8" column
    text_labels = data['question'].dropna().values
    print(text_labels)
    print("DATA EVALUATED")

    # Convert the list of embeddings into a numpy array
    embeddings_array_stacked = np.stack(embeddings.values)
    print("ARRAY CREATED")
    print(embeddings_array_stacked)
    center_embedding = np.mean(embeddings_array_stacked, axis=0)

    # Step 3: Calculate the Euclidean distance of each embedding to the center embedding
    distances = np.sqrt(np.sum((embeddings_array_stacked - center_embedding)**2, axis=1))
    cosine_distances = np.array([cosine(embedding, center_embedding) for embedding in embeddings_array_stacked])
    single_value_distance_cosine = sum(cosine_distances)/len(cosine_distances)
    single_value_distance_euc = sum(distances)/len(distances)


    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_array_stacked)
    print("ARRAY SCALED")

    # Initialize variables for silhouette analysis
    best_num_clusters = 0
    best_silhouette_score = -1
    silhouette_scores = []
    range_of_clusters = range(2, 20)  # Starting from 2 because silhouette score cannot be calculated for a single cluster

    # Calculate silhouette scores for different numbers of clusters
    for k in range_of_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
        silhouette_avg = silhouette_score(embeddings_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
        
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = k

    print("Best silhouette score was", best_silhouette_score, "for", best_num_clusters, "clusters.")

    # Plotting the silhouette scores for visual inspection
    plt.figure(figsize=(8, 4))
    plt.plot(range_of_clusters, silhouette_scores, '-o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Various Numbers of Clusters')
    plt.savefig(f'clusters-{visualization_type}.png')

    # Apply UMAP reduction and clustering with the optimal number of clusters
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine')
    embeddings_2d_umap_cosine = reducer.fit_transform(embeddings_scaled)
    print("REDUCED DIMENSIONS")

    kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_2d_umap_cosine)

    # Create dict with keys as cluster number and value of list of strings.
    class_dict = {}
    for cls, string in zip(clusters, text_labels):
        if cls not in class_dict:
            class_dict[cls] = []
        class_dict[cls].append(string)


    labels = {}
    for key, value in class_dict.items():
        completion = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": f"""Brugeren vil give dig en masse tekster. 
                 - Du skal med MAKSIMALT 3 ord give en årsag til problemerne.
                 - Dit svar skal bruges til at klassificere teksterne. 
                 - Svaret må ikke være tæt på en af disse '{', '.join(labels.values())}'.
                 - Teksterne er relateret til dette spørgsmål: {question}"""},
                {"role": "user", "content": "---\n".join(value)}
            ]
        )

        print(completion.choices[0].message.content)
        labels[key] = completion.choices[0].message.content

    # Access the centroids
    centroids = kmeans.cluster_centers_


    print(labels)
    # Visualization
    plt.figure(figsize=(10, 8))
    for i, centroid in enumerate(centroids):
        cluster_label = labels.get(i, 'N/A')  # Use the cluster label, or 'N/A' if not found
        plt.annotate(cluster_label, (centroid[0], centroid[1]), fontsize=9, weight='bold', color='black', alpha=0.75)

    plt.scatter(embeddings_2d_umap_cosine[:, 0], embeddings_2d_umap_cosine[:, 1], c=clusters, cmap='Spectral')
    # for i, txt in enumerate(text_labels):
    #     if i % 20 == 0:  # Adjust this value as needed
    #         plt.annotate(txt, (embeddings_2d_umap_cosine[i, 0], embeddings_2d_umap_cosine[i, 1]), fontsize=9, alpha=0.75, color='red')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black', label='Centroids')
    plt.title(f'2D visualisering af {question}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig(f'2D_embeddings_plot_with_optimal_clusters-{visualization_type}.png')
    plt.close()
    print("File saved with", best_num_clusters, "clusters.")

    sorted_labels = [labels[i] for i in sorted(labels.keys())]

    # Calculate the distances between centroids as before
    distances = cdist(centroids, centroids, 'euclidean')
    matrix_sum = np.sum(distances)
    disagreement_value = matrix_sum/best_num_clusters

    # Convert the distances matrix to a DataFrame for easier handling
    distances_df = pd.DataFrame(distances)

    # Label the rows and columns with the sorted_labels for clarity
    distances_df.columns = sorted_labels
    distances_df.index = sorted_labels

    # Save the DataFrame to a CSV file
    distances_df.to_csv(f'cluster-distances-{visualization_type}.csv')
    print(f"Cluster distances with named labels saved to cluster-distances-{visualization_type}.csv.")


    print("***"*5+"Stastics"+"***"*5)
    print("Single value distance cosine:", single_value_distance_cosine)
    print("Single value distance euc: ", single_value_distance_euc)
    print(f"Intercluster-disagreement value: {disagreement_value}")
