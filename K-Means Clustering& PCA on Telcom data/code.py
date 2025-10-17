import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df=pd.read_csv('dataset.csv')
print(df)
print(df.shape)
df.head()

#PART A

# Drop ID-like columns if present
id_cols = [col for col in df.columns if 'id' in col.lower()]
df = df.drop(columns=id_cols, errors='ignore')

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Fill missing values
df = df.fillna(df.median(numeric_only=True))

# Standardize numeric values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

#PART B 

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", pca.explained_variance_ratio_.sum())

# Plot PCA scatter
plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], s=30, alpha=0.6)
plt.title("PCA - 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#PART C 

inertia = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_pca, kmeans.labels_))

# Elbow Method
plt.plot(K_range, inertia, 'o-')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

# Silhouette Scores
plt.plot(K_range, sil_scores, 'o-', color='green')
plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()

# Choose K (based on plots, suppose K=3)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_pca)

# Plot clusters
plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster'], cmap='tab10', s=40)
plt.title(f"K-Means Clusters (K={k}) on PCA Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#PART D

cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print(cluster_summary)

# Example: compare churn %
if 'Churn' in df.columns:
    churn_rate = df.groupby('Cluster')['Churn'].mean()
    print("\nChurn rate per cluster:")
    print(churn_rate)

# Visualize cluster differences
sns.heatmap(cluster_summary.T, cmap="YlGnBu", annot=True)
plt.title("Cluster Profiles")
plt.show()
