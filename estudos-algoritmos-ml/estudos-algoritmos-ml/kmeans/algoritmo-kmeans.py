import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    """Seleciona K pontos aleatórios do dataset como centroides iniciais."""
    n_samples = X.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    return centroids

def assign_clusters(X, centroids):
    """Atribui cada ponto ao cluster mais próximo."""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels

def update_centroids(X, cluster_labels, k):
    """Recalcula os centroides como a média dos pontos em cada cluster."""
    new_centroids = np.array([X[cluster_labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(X, k, max_iters=100, tol=1e-4):
    """Executa o algoritmo K-Means."""
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        cluster_labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, cluster_labels, k)
         
        # Verifica convergência
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
     
    return centroids, cluster_labels

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "ecommerce_transactions.csv"

# Load the latest version
hf_dataset = kagglehub.load_dataset(
  KaggleDatasetAdapter.HUGGING_FACE,
  "pratikprasad18/e-commerce-transactions",
  file_path)

colunas_desejadas = ["Quantity", "Discount"]

# 1. Selecionar apenas as colunas desejadas
dataset_filtrado = hf_dataset.remove_columns([col for col in hf_dataset.column_names if col not in colunas_desejadas])

# 2. Remover a primeira linha (linha 0)
dataset_final = dataset_filtrado.select(range(1, len(dataset_filtrado)))
df = dataset_final.to_pandas()
X = df[['Quantity', 'Discount']].values

k = 3  # Número de clusters
centroids, cluster_labels = kmeans(X, k)
 
# Visualização dos Clusters
for i in range(k):
    plt.scatter(X[cluster_labels == i, 0], X[cluster_labels == i, 1], label=f'Cluster {i + 1}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroides')
plt.xlabel('Quantity')
plt.ylabel('Discount')
plt.title('Clusters Formados')
plt.legend()
plt.show()





#Estudos baseados no algoritmo de: https://iacomcafe.com.br/machine-learning-kmeans-python-scratch/