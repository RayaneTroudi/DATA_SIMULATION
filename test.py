import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# Charger le jeu de données Iris
iris = datasets.load_iris()
data = iris.data
target = iris.target

# Centrer les données
mean = np.mean(data, axis=0)
centered_data = data - mean

# Appliquer PCA avec trois composantes principales
pca = PCA(n_components=4)
pca_result = pca.fit_transform(centered_data)
inertia_ratios = pca.explained_variance_ratio_

# Afficher les coefficients d'inertie
print("Coefficients d'inertie de chaque axe:")
for i, ratio in enumerate(inertia_ratios):
    print(f"Axe {i + 1}: {ratio:.4f}")
# Tracer les données dans l'espace de PC1 et PC3
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=target, cmap='viridis', s=50)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection sur PC1 et PC2 après PCA')
plt.colorbar(scatter, label='Classe')
plt.show()
