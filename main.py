import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# Function for curve fitting
def model_function(x, *params):
    # Choose a simple model function, for example, exponential growth
    return params[0] * np.exp(params[1] * x)

# Function to estimate lower and upper limits of the confidence range
def err_ranges(popt, pcov, x):
    perr = np.sqrt(np.diag(pcov))
    lower_bound = model_function(x, *popt) - 1.96 * perr[0]
    upper_bound = model_function(x, *popt) + 1.96 * perr[0]
    return lower_bound, upper_bound

# Generate or load your data
# Replace this with your actual data
data = pd.DataFrame({
    'Country': ['Country1', 'Country2', 'Country3', 'Country4'],
    'GDP_Per_Capita': [10000, 15000, 8000, 12000],
    'CO2_Per_Head': [5.5, 8.2, 4.0, 6.5],
    'Year': [2010, 2015, 2020, 2025]
})

# Student ID
student_id = 22092233

# Normalize your data if needed

# Perform clustering using KMeans
num_clusters = 3  # Set the number of clusters
features = ['GDP_Per_Capita', 'CO2_Per_Head']
X = data[features].values
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Plot cluster membership
plt.scatter(data['GDP_Per_Capita'], data['CO2_Per_Head'], c=data['Cluster'], cmap='coolwarm', label='Clusters')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='blue', label='Cluster Centers')
plt.xlabel('GDP Per Capita')
plt.ylabel('CO2 Per Head')
plt.title('Clustering Results\nStudent ID: {}'.format(student_id))
plt.legend()
plt.show()

# Fit a model using curve_fit
x_values = data['Year']
y_values = data['GDP_Per_Capita']

# Provide initial guesses for parameters
initial_guess = [y_values.iloc[0], 0.1]

popt, pcov = curve_fit(model_function, x_values, y_values, p0=initial_guess)

# Generate predictions for the next 10 years
future_years = np.arange(2026, 2036, 1)
predicted_values = model_function(future_years, *popt)
lower_bound, upper_bound = err_ranges(popt, pcov, future_years)

# Plot the best fitting function and confidence range
plt.plot(x_values, y_values, 'o', label='Actual Data')
plt.plot(future_years, predicted_values, label='Best Fitting Function', color='purple')
plt.fill_between(future_years, lower_bound, upper_bound, color='purple', alpha=0.3, label='Confidence Range')
plt.xlabel('Year')
plt.ylabel('GDP Per Capita')
plt.title('Curve Fitting Results\nStudent ID: {}'.format(student_id))
plt.legend()
plt.show()
