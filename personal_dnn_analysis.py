import pandas as pd  # Data handling
import numpy as np   # Numerical calculations
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns  # Statistical visualization
from sklearn.decomposition import PCA  # Principal Component Analysis
from sklearn.preprocessing import StandardScaler  # Standardizing data

# Load the CSV file
file_path = "output/summary_file_fixed.csv"  # Use relative path
df = pd.read_csv(file_path)

# Display basic info about the dataset
print(df.info())  # Shows data types and missing values
print(df.head())  # Shows the first few rows
