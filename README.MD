# Image Clustering Tool

A powerful tool for unsupervised clustering of images using state-of-the-art deep learning models. This project extracts features from images using pre-trained models like CLIP and ResNet, reduces dimensions with UMAP or PCA, and then applies clustering algorithms like HDBSCAN or K-means.

## Features

- **Multiple Feature Extractors**: 
  - CLIP (Contrastive Language-Image Pre-Training)
  - ResNet (18, 34, 50, or 101 variants)
  
- **Dimension Reduction Techniques**:
  - UMAP (Uniform Manifold Approximation and Projection)
  - PCA (Principal Component Analysis)
  
- **Clustering Algorithms**:
  - HDBSCAN (Hierarchical Density-Based Spatial Clustering)
  - K-means

- **Visualization**: Automatic generation of 2D plots showing the clustering results

## Installation

```bash
pip install torch torchvision pillow transformers umap-learn hdbscan scikit-learn matplotlib
```

## Usage

1. Prepare your configuration in a JSON file (see examples below)
2. Run the clustering script:

```bash
python3 cluster.py --config config.json
```

## Configuration

The tool uses JSON configuration files. Here's an example:

```json
[
  {
    "run_name": "clip_umap_hdbscan",
    "feature_extractor": "clip",
    "clip_model_name": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "dim_reduction_method": "umap",
    "dim_reduction_components": 2,
    "cluster_method": "hdbscan",
    "hdbscan_min_cluster_size": 2,
    "folder_path": "Clustering",
    "show_plots": true,
    "save_plots": true
  }
]
```

### Configuration Parameters

- `run_name`: Name of the clustering run (used for output files)
- `feature_extractor`: Model to use for feature extraction (`clip` or `resnet`)
- `clip_model_name`: When using CLIP, the specific model to use
- `resnet_variant`: When using ResNet, the variant to use (18, 34, 50, 101)
- `dim_reduction_method`: Method for dimension reduction (`umap` or `pca`)
- `dim_reduction_components`: Number of dimensions to reduce to (2 for visualization)
- `cluster_method`: Clustering algorithm to use (`hdbscan` or `kmeans`)
- `hdbscan_min_cluster_size`: Minimum size of clusters when using HDBSCAN
- `kmeans_n_clusters`: Number of clusters for K-means
- `folder_path`: Path to the folder containing images to cluster
- `show_plots`: Whether to display plots during runtime (default: `true`)
- `save_plots`: Whether to save plots to files (default: `true`)

## Output

The tool creates a directory structure with:
- Visualizations of the clustering results
- Lists of images in each cluster
- Embeddings and reduced-dimension data for further analysis

## Examples

### Using CLIP with UMAP and HDBSCAN

```json
{
  "run_name": "clip_umap_hdbscan",
  "feature_extractor": "clip",
  "clip_model_name": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
  "dim_reduction_method": "umap",
  "dim_reduction_components": 2,
  "cluster_method": "hdbscan",
  "hdbscan_min_cluster_size": 2,
  "folder_path": "images/photos",
  "show_plots": true,
  "save_plots": true
}
```

### Using ResNet with PCA and K-means

```json
{
  "run_name": "resnet50_pca_kmeans",
  "feature_extractor": "resnet",
  "resnet_variant": "resnet50",
  "dim_reduction_method": "pca",
  "dim_reduction_components": 2,
  "cluster_method": "kmeans",
  "kmeans_n_clusters": 5,
  "folder_path": "images/products",
  "show_plots": false,
  "save_plots": true
}
```

## License

MIT

## Contributing

Contributions, issues, and feature requests are welcome!
