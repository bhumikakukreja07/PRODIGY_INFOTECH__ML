import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("Mall_Customers.csv").drop(columns=['CustomerID', 'Gender'])
scaled = StandardScaler().fit_transform(df)

# Elbow method to find optimal K
inertia = [KMeans(n_clusters=k, random_state=42).fit(scaled).inertia_ for k in range(1, 11)]
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method'); plt.xlabel('Clusters'); plt.ylabel('Inertia'); plt.show()

# KMeans with optimal clusters
kmeans = KMeans(n_clusters=5, random_state=42).fit(scaled)
df['Cluster'] = kmeans.labels_

# Show cluster means
print(df.groupby('Cluster').mean())