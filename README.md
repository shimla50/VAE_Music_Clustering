# VAE Music Clustering Project

This project implements an unsupervised learning pipeline for clustering hybrid language music tracks using Variational Autoencoders (VAE).

## Project Structure

- `data/`: Stores the datasets (audio and lyrics).
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis and visualizations.
- `src/`: Contains Python scripts for VAE implementation, data processing, clustering, and evaluation.
- `results/`: Stores output results such as latent space visualizations and clustering metrics.

## Requirements

To install the required libraries, run:
```bash
pip install -r requirements.txt
 
## Dataset

This project uses the Hugging Face dataset: **ccmusic-database/music_genre**  
Dataset page: https://huggingface.co/datasets/ccmusic-database/music_genre

The dataset is not included in this GitHub repository because it is large.
It will be downloaded automatically when running the scripts below.
