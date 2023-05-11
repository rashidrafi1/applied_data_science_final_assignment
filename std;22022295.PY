import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score

# Read the dataset
df_cluster = pd.read_csv('API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_5447813.csv', skiprows=4)

# Display the head of the dataset
df_cluster.head()


df_cluster.describe()

df_cluster.info()


# Load the dataset
df = pd.read_csv("API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_5447813.csv", skiprows=4)

# Select relevant columns and drop missing values
df_cleaned = df[["Country Name", "2021"]].dropna()

# Rename columns
df_cleaned.columns = ["Country", "Inflation"]

# Set index to country name
df_cleaned.set_index("Country", inplace=True)

# Remove rows with invalid inflation values (negative or zero)
df_cleaned = df_cleaned[df_cleaned["Inflation"] > 0]

# Log-transform the inflation values to reduce skewness
df_cleaned["Inflation"] = np.log(df_cleaned["Inflation"])

# Standardize the data using z-score normalization
df_cleaned = (df_cleaned - df_cleaned.mean()) / df_cleaned.std()

# Save the cleaned dataset to a new file
df_cleaned.to_csv("clustering_dataset.csv")


# Load the cleaned dataset
df = pd.read_csv("clustering_dataset.csv")

# Select 4 African countries
african_countries = ["Kenya", "South Africa", "Nigeria", "Angola"]
df_african = df.loc[df["Country"].isin(african_countries)]

# Calculate the percentage of inflation for each country
inflation_percentages = df_african["Inflation"].apply(lambda x: x * 100)

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(inflation_percentages, labels=df_african["Country"], autopct='%1.1f%%', startangle=90)
plt.title("Inflation of Sub-saharan African Countries")
plt.axis('equal')
plt.show()


# Load the cleaned dataset
df = pd.read_csv("clustering_dataset.csv")

# Extract the Inflation column and normalize the data
X = df['Inflation'].values.reshape(-1, 1)
X_norm = (X - X.mean()) / X.std()

# Define the range of number of clusters to try
n_clusters_range = range(2, 11)

# Initialize an empty list to store the inertia values
inertia_values = []

# Iterate over the number of clusters and compute the inertia
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_norm)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(n_clusters_range, inertia_values)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Plot: Optimal Number of Clusters")
plt.show()




# Load the cleaned dataset
df = pd.read_csv('clustering_dataset.csv')

# Extract Inflation column and normalize
X = df['Inflation'].values.reshape(-1, 1)
X_norm = StandardScaler().fit_transform(X)

# Perform Gaussian Mixture Model clustering with n_clusters=3
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_norm)
df['Cluster'] = gmm.predict(X_norm)

# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    scatter = ax.scatter(cluster_data.index, cluster_data['Inflation'],
                         color=colors[i], label=f'Cluster {i+1}')
plt.xticks(np.arange(0, df.shape[0], 50), np.arange(0, df.shape[0], 50), fontsize=12)
plt.xlabel('Country Index', fontsize=14)
plt.ylabel('Inflation consumer prices (annual %)', fontsize=14)
plt.title('Gaussian Mixture Model Clustering plot', fontsize=16)
ax.legend(fontsize=12)

# Add annotation for the cluster centers
centers = gmm.means_
for i, center in enumerate(centers):
    ax.annotate(f'Cluster {i+1} center: {center[0]:,.2f}', xy=(i+1, center[0]), xytext=(6, 0),
                textcoords="offset points", ha='left', va='center', fontsize=12, color=colors[i])

plt.show()

# print countries in each cluster in a table
for i in range(4):
    print(f'Cluster {i+1}:')
    cluster_data = df[df['Cluster']==i]
    cluster_table = pd.DataFrame({'Country': cluster_data['Country'].values})
    print(cluster_table)
    
    
# Read the dataset
df_FIT = pd.read_csv('API_NY.GDP.DEFL.KD.ZG_DS2_en_csv_v2_5358490.csv', skiprows=4)

# Display the head of the dataset
df_FIT.head()


df_FIT.describe()

df_FIT.info()


# Load the dataset into a pandas DataFrame
df = pd.read_csv('API_NY.GDP.DEFL.KD.ZG_DS2_en_csv_v2_5358490.csv', skiprows=4)

# Select only the necessary data for fitting analysis
df = df[['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', *df.columns[-32:-1]]]  # Select the columns for analysis

# Rename columns to simpler names
df.columns = ['Country', 'Code', 'Indicator', 'IndicatorCode', *range(1990, 2021)]  # Rename columns for easier reference

# Melt the DataFrame to transform the columns into rows
df_melted = pd.melt(df, id_vars=['Country', 'Code', 'Indicator', 'IndicatorCode'], var_name='Year', value_name='Value')  # Reshape data into long format

# Drop rows with missing values
df_cleaned = df_melted.dropna()  # Remove rows with missing values

# Save the cleaned data to a new CSV file
df_cleaned.to_csv('fitting_data.csv', index=False)  # Save the cleaned data to a new CSV file without row indices




# Load the cleaned dataset
df = pd.read_csv('fitting_data.csv')

# Filter data for Nigeria
nigeria_data = df[df['Country'] == 'Nigeria']

# Extract the necessary columns
years = nigeria_data['Year'].values
values = nigeria_data['Value'].values

# Fit a polynomial curve to the data
coeffs = np.polyfit(years, values, deg=2)
poly_func = np.poly1d(coeffs)

# Calculate the residuals
residuals = values - poly_func(years)

# Calculate the standard deviation of the residuals
std_dev = np.std(residuals)

# Generate predictions for future years
future_years = np.arange(years.min(), years.max() + 21)  # Predict for 20 additional years
predicted_values = poly_func(future_years)

# Calculate upper and lower confidence bounds
upper_bound = predicted_values + 2 * std_dev
lower_bound = predicted_values - 2 * std_dev

# Plot the best fitting function and confidence range
plt.figure(figsize=(12, 8))
plt.plot(years, values, 'ko', label='Actual Data')
plt.plot(future_years, predicted_values, 'r-', label='Best Fitting Function')
plt.fill_between(future_years, lower_bound, upper_bound, color='gray', alpha=0.4, label='Confidence Range')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Inflation Value', fontsize=14)
plt.title('Polynomial Model Fit for Nigeria', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()




