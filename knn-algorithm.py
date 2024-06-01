import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load Data
    """
    data = pd.read_csv(file_path)
    return data[['BodyweightKg', 'TotalKg']].dropna()

def classify(row):
    """
    Classification Data
    """
    ratio = row['TotalKg'] / row['BodyweightKg']
    if ratio >= 6.75:
        return 'Pro'
    elif ratio >= 4.72:
        return 'Amateur'
    else:
        return 'Beginner'

def visualize_data(data, new_data=None):
    """
    Visualization Data
    """
    # Define New Class
    data['Class'] = data.apply(classify, axis=1)

    # Mapping color
    color_mapping = {'Pro': 'red', 'Amateur': 'blue', 'Beginner': 'green'}
    data['Color'] = data['Class'].map(color_mapping)

    # Scatterplot visualization
    plt.figure(figsize=(10, 6))
    for class_type in ['Pro', 'Amateur', 'Beginner']:
        subset = data[data['Class'] == class_type]
        plt.scatter(subset['BodyweightKg'], subset['TotalKg'], 
                    label=class_type, color=color_mapping[class_type], alpha=0.6)

    # Add new data
    if new_data is not None:
        plt.scatter(new_data['BodyweightKg'], new_data['TotalKg'], color='black', label='New Data', marker='x')

    plt.xlabel('BodyweightKg')
    plt.ylabel('TotalKg')
    plt.title('BodyweightKg vs TotalKg (Classified)')
    plt.legend()

    plt.show()

def classify_new_data(new_data):
    """
    A function that classifies new data in a scatterplot

    """
    new_ratio = new_data['TotalKg'] / new_data['BodyweightKg']
    if new_ratio.values >= 6.75:
        return "Pro"
    elif new_ratio.values >= 4.72:
        return "Amateur"
    else:
        return "Beginner"

# Load data
data = load_data("openpowerlifting.csv")

# Add new data
new_data = pd.DataFrame({'BodyweightKg': [65], 'TotalKg': [210]})
visualize_data(data, new_data=new_data)

# Classification new data
print("New Data is classified as:", classify_new_data(new_data))
