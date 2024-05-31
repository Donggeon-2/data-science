import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read Dataset
df = pd.read_csv('openpowerlifting.csv')

# Select the required feature (Age, Sex, TotalKg)
selected_features = ['Age', 'Sex', 'TotalKg']
df_selected = df[selected_features]

# Binaryize Sex (Male: 1, Female: 0)
df_selected['Sex'] = df_selected['Sex'].map({'M': 1, 'F': 0})

# Processing missing values (replaced by average)
df_selected.fillna(df_selected.mean(), inplace=True)

# Feature Scaling (Standardization)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_scaled)

# Check cluster assignment results
df_selected['Cluster'] = kmeans.labels_

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'y', 'c', 'm']
for cluster in df_selected['Cluster'].unique():
    cluster_data = df_selected[df_selected['Cluster'] == cluster]
    ax.scatter(cluster_data['Age'], cluster_data['Sex'], cluster_data['TotalKg'], c=colors[cluster], label=f'Cluster {cluster}')

ax.set_title('K-Means Clustering of Age, Sex, and Total Weight')
ax.set_xlabel('Age')
ax.set_ylabel('Sex (0: Female, 1: Male)')
ax.set_zlabel('Total Weight (Kg)')
plt.legend(title='Cluster', loc='upper left')
plt.show()
