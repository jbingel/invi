import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
from scipy.spatial.distance import cdist, cosine

from text_processing import get_cluster_label


def create_visualizations(
        df: pd.DataFrame, 
        embeddings: np.ndarray, 
        output_dir: str,
        response_type: str,
        text_labels: list[str] = None,
        question: str = None
    ):
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
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
    kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
    cluster_index = kmeans.fit_predict(embeddings_scaled)
    df['cluster_index'] = cluster_index


    # Plotting the silhouette scores for visual inspection
    plt.figure(figsize=(8, 4))
    plt.plot(range_of_clusters, silhouette_scores, '-o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Various Numbers of Clusters')
    plt.savefig(f'{output_dir}/clusters-{response_type}.png')
    # Apply UMAP reduction and
    #  clustering with the optimal number of clusters
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine')
    embeddings_2d_umap_cosine = reducer.fit_transform(embeddings_scaled)
    print("REDUCED DIMENSIONS")

    kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_2d_umap_cosine)

    # Create dict with keys as cluster number and value of list of strings.
    class_dict = {}
    for cls, responses in zip(clusters, text_labels):
        if cls not in class_dict:
            class_dict[cls] = []
        class_dict[cls].append(responses)

    # Create labels for each of the clusers.
    labels = {}
    for key, responses in class_dict.items():
        labels[key] = responses[0]
        # labels[key] = get_cluster_label(question, tuple(responses), tuple(labels.values()))
        
    
    def map_cluster_to_label(cluster_index):
        return labels.get(cluster_index, 'N/A')  # Return 'N/A' if no label is found for the cluster index

    df['cluster_label'] = df['cluster_index'].apply(map_cluster_to_label)

    df.to_excel(f"{output_dir}/{response_type}-final.xlsx", index=False)

    
    # Access the centroids
    centroids = kmeans.cluster_centers_

    # Visualization
    plt.figure(figsize=(10, 8))
    for i, centroid in enumerate(centroids):
        cluster_label = labels.get(i, 'N/A')  # Use the cluster label, or 'N/A' if not found
        plt.annotate(cluster_label, (centroid[0], centroid[1]), fontsize=9, weight='bold', color='black', alpha=0.75)

    plt.scatter(embeddings_2d_umap_cosine[:, 0], embeddings_2d_umap_cosine[:, 1], c=clusters, cmap='Spectral')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black', alpha=.5, label='Centroids')
    plt.title(f'2D visualisering af {question}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig(f'{output_dir}/2D_embeddings_plot_with_optimal_clusters-{response_type}.png')
    plt.close()
    print("File saved with", best_num_clusters, "clusters.")
