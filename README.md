# K-Means and GMM File Clustering and Archiving

A Python script that uses k-means and gmm clustering to intelligently group files based on their characteristics and creates compressed tar archives for each cluster.

## Features

- **Intelligent Clustering**: Groups files based on:
  - File size
  - File type (MIME type)
  - Modification time
  - File extension

- **Flexible Compression**: Supports:
  - XZ compression (tar.xz) with levels 0-9
  - Zstandard compression (tar.zst) with levels 1-22
  - Both simultaneously
  - Configurable compression levels for optimal speed/size trade-off

- **Recursive Scanning**: Can scan subdirectories
- **Detailed Statistics**: Shows cluster information and file counts
- **Preserves Structure**: Maintains relative paths in archives

Command-Line Arguments

#### Required Arguments
- `-i, --input`: Input directory containing files to cluster
- `-o, --output`: Output directory for compressed archives

#### Optional Arguments
- `-k, --clusters`: Number of clusters (default: 5)
- `-c, --compression`: Compression type - `xz`, `zstd`, or `both` (default: xz)
- `-r, --recursive`: Scan subdirectories recursively
- `-a, --algorithm`: Set cluster algorithm to use (default: kmeans)
- `--xz-level`: XZ compression level 0-9 (default: 6)
  - 0 = fastest, lowest compression
  - 6 = balanced (default)
  - 9 = slowest, best compression
- `--zstd-level`: Zstandard compression level 1-22 (default: 3)
  - 1 = fastest, lowest compression
  - 3 = balanced (default)
  - 10 = good compression with reasonable speed
  - 19 = high compression
  - 20-22 = ultra compression (very slow)
## Installation

### Quick Setup (Recommended)

Run the automated setup script:


```bash
pip install -r requirements.txt
```

## Usage examples
```bash
python cluster_and_compress.py -i ./project -o ./backup -k 8 -c zstd -r

# XZ fast compression (level 1)
python cluster_and_compress.py -i ~/Documents -o ~/Backup -k 8 -c xz -a gmm --xz-level 1

# Zstandard fast compression (level 1)
python cluster_and_compress.py -i ~/Documents -o ~/Backup -k 8 -c zstd -a kmeans --zstd-level 1
```	
