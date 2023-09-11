import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_data(file_path):
    pattern = re.compile(
        r"""
        (\s+\d+
        \s+\d\.\d+e[+-]\d+
        \s+\d\.\d+e[+-]\d+
        \s+\d\.\d+e[+-]\d+
        \s+(?P<max_f_devi>\d\.\d+e[+-]\d+)
        \s+(?P<min_f_devi>\d\.\d+e[+-]\d+)
        \s+(?P<avg_f_devi>\d\.\d+e[+-]\d+)
        \n
        )
        """, re.VERBOSE)
    
    with open(file_path, 'r') as file:
        content = file.read()
        
    matches = pattern.finditer(content)
    max_f_devi_values = [float(match.group('max_f_devi')) for match in matches]
    
    return max_f_devi_values

# Find all files that match the pattern "model_devi_*K"
file_paths = glob.glob('model_devi_Sn.out')

# Extract the max_f_devi data from each file and print the data
data = []
for file_path in file_paths:
    max_f_devi_values = extract_data(file_path)
    data.append(max_f_devi_values)
    print(f"Data from {file_path}: {max_f_devi_values}")

# Compute the total number of data points
total_data_points = sum(len(max_f_devi_values) for max_f_devi_values in data)
print(f"Total number of data points: {total_data_points}")

# Plot the probability density function for each data set
plt.figure(figsize=(10, 6))
for i, max_f_devi_values in enumerate(data):
    # Skip the dataset if it has zero variance
    if np.var(max_f_devi_values) == 0:
        print(f"Skipping {file_paths[i]} because its data has zero variance.")
        continue
    sns.kdeplot(max_f_devi_values, label=file_paths[i])

plt.legend()
plt.xlabel('max_f_devi')
plt.ylabel('Density')
plt.title('Probability Density Function of max_f_devi from different files')
plt.savefig('model_devi.tiff')
plt.show()

