import json
import matplotlib.pyplot as plt
import re
import os

# Load the data from the JSON file
json_file_path = 'results/direct_vision_all_images/gpt_vision_clusters.json'

# Function to extract clusters from the analysis_process text
def extract_clusters_from_text(text):
    # Pattern to match the cluster sections
    pattern = r'"name": "([^"]+)",[^}]*"description": "([^"]+)",[^}]*"reasoning": "([^"]+)",[^}]*"images": \[(.*?)\]'
    matches = re.findall(pattern, text)
    
    clusters = []
    for match in matches:
        name, description, reasoning, images_str = match
        # Extract image names
        image_pattern = r'"([^"]+)"'
        images = re.findall(image_pattern, images_str)
        
        clusters.append({
            "name": name,
            "description": description,
            "reasoning": reasoning,
            "images": images
        })
    
    return clusters

# Load the JSON data
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Extract clusters from the analysis_process
analysis_process = data['analysis_process']
clusters = extract_clusters_from_text(analysis_process)

# Create visualizations
def create_visualizations(clusters):
    # 1. Pie chart of cluster sizes
    plt.figure(figsize=(10, 6))
    labels = [f"{cluster['name']} ({len(cluster['images'])})" for cluster in clusters]
    sizes = [len(cluster['images']) for cluster in clusters]
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Architectural Plans by Cluster Type')
    
    # Save the pie chart
    plt.savefig('results/direct_vision_all_images/cluster_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar chart of cluster sizes
    plt.figure(figsize=(12, 6))
    plt.bar(labels, sizes, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Floor Plans')
    plt.title('Number of Floor Plans in Each Cluster')
    plt.tight_layout()
    
    # Save the bar chart
    plt.savefig('results/direct_vision_all_images/cluster_distribution_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create a text summary file
    with open('results/direct_vision_all_images/cluster_summary.txt', 'w') as f:
        f.write("# Architectural Floor Plan Clustering Summary\n\n")
        
        for i, cluster in enumerate(clusters, 1):
            f.write(f"## Cluster {i}: {cluster['name']}\n\n")
            f.write(f"**Description:** {cluster['description']}\n\n")
            f.write(f"**Reasoning:** {cluster['reasoning']}\n\n")
            f.write(f"**Number of Images:** {len(cluster['images'])}\n\n")
            f.write("**Images:** " + ", ".join(cluster['images']) + "\n\n")
            f.write("-" * 50 + "\n\n")

# Create the visualizations
create_visualizations(clusters)

print("Visualizations created successfully in the results/direct_vision_all_images directory:")
print("1. cluster_distribution_pie.png - Pie chart of cluster distribution")
print("2. cluster_distribution_bar.png - Bar chart of cluster sizes")
print("3. cluster_summary.txt - Detailed text summary of clusters") 