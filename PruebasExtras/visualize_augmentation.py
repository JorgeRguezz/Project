import matplotlib.pyplot as plt
import numpy as np
import json
from datasets import load_dataset
import pandas as pd
import seaborn as sns

# Load the original dataset
print("Loading original MMLU-Pro dataset...")
original_dataset = load_dataset("TIGER-Lab/MMLU-Pro")

# Load the augmented dataset - adjust the path as needed
# This assumes you have the augmented dataset saved somewhere
print("Loading augmented dataset...")
# Option 1: If saved as a Hugging Face dataset
# augmented_dataset = load_dataset("path_to_augmented_dataset")

# Option 2: If saved as JSON or other format
# You'll need to adjust this based on how you saved your augmented data
try:
    with open('c:\\Users\\rrjor\\Desktop\\UPM\\TFG\\Project\\augmented_dataset.json', 'r') as f:
        augmented_data = json.load(f)
except FileNotFoundError:
    print("Augmented dataset file not found. Please adjust the path.")
    # For demonstration, we'll create dummy data
    augmented_data = {"train": original_dataset["test"]}  # Just a placeholder

# Count categories in original dataset
original_categories = {}
for item in original_dataset["test"]:
    category = item["category"]
    if category in original_categories:
        original_categories[category] += 1
    else:
        original_categories[category] = 1

# Count categories in augmented dataset
# Adjust this based on your actual augmented data structure
augmented_categories = {}
try:
    for item in augmented_data["train"]:  # Adjust key if needed
        category = item["category"]
        if category in augmented_categories:
            augmented_categories[category] += 1
        else:
            augmented_categories[category] = 1
except (KeyError, TypeError):
    print("Could not process augmented dataset. Using dummy data for visualization.")
    # Create dummy augmented data with 2x the original count
    augmented_categories = {k: v*2 for k, v in original_categories.items()}

# Create a DataFrame for visualization
categories = list(set(list(original_categories.keys()) + list(augmented_categories.keys())))
data = []

for category in categories:
    orig_count = original_categories.get(category, 0)
    aug_count = augmented_categories.get(category, 0)
    data.append({
        'Category': category,
        'Original': orig_count,
        'Augmented': aug_count,
        'Added': aug_count - orig_count
    })

df = pd.DataFrame(data)

# Sort by number of examples
df = df.sort_values('Original', ascending=False)

# Create the visualization
plt.figure(figsize=(15, 10))

# Plot stacked bars
ax = sns.barplot(x='Category', y='Original', data=df, color='blue', label='Original')
sns.barplot(x='Category', y='Added', data=df, color='orange', label='Added by Augmentation', bottom=df['Original'])

# Customize the plot
plt.title('Distribution of Examples by Category: Original vs. Augmented', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Number of Examples', fontsize=14)
plt.xticks(rotation=90)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the visualization
plt.savefig('c:\\Users\\rrjor\\Desktop\\UPM\\TFG\\Project\\augmentation_distribution.png', dpi=300, bbox_inches='tight')

# Calculate and display augmentation statistics
total_original = sum(original_categories.values())
total_augmented = sum(augmented_categories.values())
increase = total_augmented - total_original
percent_increase = (increase / total_original) * 100

print(f"Original dataset size: {total_original} examples")
print(f"Augmented dataset size: {total_augmented} examples")
print(f"Additional examples: {increase}")
print(f"Percentage increase: {percent_increase:.2f}%")

# Create a second visualization showing percentage increase by category
plt.figure(figsize=(15, 8))

# Calculate percentage increase for each category
df['Percent Increase'] = (df['Augmented'] / df['Original'] - 1) * 100
df_percent = df.sort_values('Percent Increase', ascending=False)

# Handle infinite values (when original count was 0)
df_percent['Percent Increase'] = df_percent['Percent Increase'].replace([np.inf, -np.inf], np.nan)
df_percent = df_percent.dropna(subset=['Percent Increase'])

# Plot percentage increase
sns.barplot(x='Category', y='Percent Increase', data=df_percent, palette='viridis')
plt.title('Percentage Increase in Examples by Category After Augmentation', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Percentage Increase (%)', fontsize=14)
plt.xticks(rotation=90)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.tight_layout()

# Save the second visualization
plt.savefig('c:\\Users\\rrjor\\Desktop\\UPM\\TFG\\Project\\augmentation_percentage.png', dpi=300, bbox_inches='tight')

plt.show()

# Save the statistics to a JSON file
stats = {
    "total_original": total_original,
    "total_augmented": total_augmented,
    "additional_examples": increase,
    "percentage_increase": percent_increase,
    "category_stats": df.to_dict(orient='records')
}

with open('c:\\Users\\rrjor\\Desktop\\UPM\\TFG\\Project\\augmentation_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

print("Statistics saved to augmentation_stats.json")