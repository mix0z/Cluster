import os
import re
import json
import math
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
output_dir = 'results/direct_vision_all_images'
clustering_folder = 'Clustering'
summary_file = os.path.join(output_dir, 'cluster_summary.txt')

# Function to ensure directory exists
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to extract cluster information from summary file
def parse_cluster_summary(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    clusters = []
    cluster_sections = re.split(r'-{10,}', content)
    
    for section in cluster_sections:
        if not section.strip():
            continue
            
        # Extract cluster information
        name_match = re.search(r'## Cluster \d+: (.+?)\n', section)
        if not name_match:
            continue
            
        name = name_match.group(1)
        
        desc_match = re.search(r'\*\*Description:\*\* (.+?)\n', section)
        description = desc_match.group(1) if desc_match else ""
        
        images_match = re.search(r'\*\*Images:\*\* (.+?)\n', section)
        if not images_match:
            continue
            
        image_str = images_match.group(1)
        images = [img.strip() for img in image_str.split(',')]
        
        clusters.append({
            'name': name,
            'description': description,
            'images': images
        })
    
    return clusters

# Get clusters from summary file
clusters = parse_cluster_summary(summary_file)

# Create gallery images for each cluster
for cluster_idx, cluster in enumerate(clusters):
    cluster_name = re.sub(r'[^\w\s-]', '', cluster["name"]).strip().replace(' ', '_')
    cluster_dir = os.path.join(output_dir, f"cluster_{cluster_idx}_{cluster_name}")
    
    # Create directory for cluster if it doesn't exist
    ensure_dir_exists(cluster_dir)
    
    # Collect image paths for this cluster
    image_paths = []
    for img_name in cluster["images"]:
        img_path = os.path.join(clustering_folder, img_name.strip())
        if os.path.exists(img_path):
            image_paths.append(img_path)
    
    if not image_paths:
        print(f"No images found for cluster {cluster_idx}: {cluster['name']}")
        continue
    
    # Create gallery image
    num_images = len(image_paths)
    cols = min(4, num_images)  # Max 4 images per row
    rows = math.ceil(num_images / cols)
    
    fig = plt.figure(figsize=(cols * 4, rows * 4))  # Larger figure for better visibility
    fig.suptitle(f"Cluster {cluster_idx+1}: {cluster['name']} ({num_images} images)", fontsize=16)
    
    # Add a subtitle with the description
    plt.figtext(0.5, 0.92, cluster["description"], wrap=True, 
                horizontalalignment='center', fontsize=10)
    
    for i, path in enumerate(image_paths, start=1):
        ax = fig.add_subplot(rows, cols, i)
        try:
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(os.path.basename(path), fontsize=9)
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            continue
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust for the main title
    
    # Save gallery image
    gallery_path = os.path.join(cluster_dir, f"gallery_cluster_{cluster_idx+1}.png")
    plt.savefig(gallery_path, dpi=200, bbox_inches='tight')
    print(f"Saved gallery for cluster {cluster_idx+1}: {cluster['name']}")
    
    # Close the figure to free memory
    plt.close()

# Create an overview of all clusters
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Architectural Floor Plan Clusters Overview", fontsize=20)

# Define grid for the overview
num_clusters = len(clusters)
cols = min(3, num_clusters)
rows = math.ceil(num_clusters / cols)

for cluster_idx, cluster in enumerate(clusters):
    # For each cluster, show just one representative image
    if not cluster["images"]:
        continue
        
    ax = fig.add_subplot(rows, cols, cluster_idx+1)
    
    # Use the first image as representative
    img_path = os.path.join(clustering_folder, cluster["images"][0].strip())
    
    try:
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Cluster {cluster_idx+1}: {cluster['name']}\n({len(cluster['images'])} images)", fontsize=10)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        continue

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the main title
overview_path = os.path.join(output_dir, "cluster_overview.png")
plt.savefig(overview_path, dpi=200, bbox_inches='tight')
print(f"Saved cluster overview to {overview_path}")
plt.close()

# Create an HTML report with all clusters and their galleries
html_report_path = os.path.join(output_dir, "cluster_report.html")
with open(html_report_path, 'w') as f:
    f.write("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Architectural Floor Plan Clustering Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 2px solid #eee;
            }
            h2 {
                color: #444;
                margin-top: 30px;
                padding-bottom: 5px;
                border-bottom: 1px solid #eee;
            }
            .cluster-card {
                margin: 20px 0;
                padding: 15px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .cluster-description {
                font-style: italic;
                color: #666;
                margin-bottom: 15px;
            }
            .gallery {
                text-align: center;
                margin: 20px 0;
            }
            .gallery img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .image-list {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 15px;
            }
            .image-item {
                border: 1px solid #eee;
                border-radius: 4px;
                padding: 5px;
                text-align: center;
                width: 150px;
            }
            .image-item img {
                max-width: 100%;
                height: auto;
            }
            .image-name {
                font-size: 0.8em;
                color: #666;
                margin-top: 5px;
            }
            .stats {
                margin: 20px 0;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            .chart {
                text-align: center;
                margin: 30px 0;
            }
            .chart img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Architectural Floor Plan Clustering Results</h1>
            
            <div class="stats">
                <h2>Cluster Statistics</h2>
                <p>Total number of clusters: <strong>""" + str(len(clusters)) + """</strong></p>
                <p>Total number of images: <strong>""" + str(sum(len(c["images"]) for c in clusters)) + """</strong></p>
            </div>
            
            <div class="chart">
                <h2>Overview</h2>
                <img src="cluster_overview.png" alt="Clusters Overview">
            </div>
            
            <div class="chart">
                <h2>Distribution</h2>
                <img src="cluster_distribution_pie.png" alt="Cluster Distribution Pie Chart">
                <img src="cluster_distribution_bar.png" alt="Cluster Distribution Bar Chart">
            </div>
    """)
    
    # Add each cluster
    for cluster_idx, cluster in enumerate(clusters):
        cluster_name = re.sub(r'[^\w\s-]', '', cluster["name"]).strip().replace(' ', '_')
        gallery_path = f"cluster_{cluster_idx}_{cluster_name}/gallery_cluster_{cluster_idx+1}.png"
        
        f.write(f"""
            <div class="cluster-card">
                <h2>Cluster {cluster_idx+1}: {cluster['name']}</h2>
                <div class="cluster-description">{cluster['description']}</div>
                <p>Number of images: <strong>{len(cluster['images'])}</strong></p>
                
                <div class="gallery">
                    <img src="{gallery_path}" alt="Gallery for Cluster {cluster_idx+1}">
                </div>
                
                <h3>Images in this cluster:</h3>
                <div class="image-list">
        """)
        
        # Add individual images
        for img_name in cluster["images"]:
            img_path = os.path.join(clustering_folder, img_name.strip())
            if os.path.exists(img_path):
                f.write(f"""
                    <div class="image-item">
                        <img src="../{img_path}" alt="{img_name}">
                        <div class="image-name">{img_name}</div>
                    </div>
                """)
        
        f.write("""
                </div>
            </div>
        """)
    
    f.write("""
        </div>
    </body>
    </html>
    """)

print(f"Generated HTML report at {html_report_path}") 