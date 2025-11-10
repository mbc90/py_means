#!/usr/bin/env python3
"""
Gaussian Mixture Models (GMM) File Clustering and Archiving Script

This script clusters files using GMM based on file characteristics
(size, type, modification time) and creates compressed tar archives for each cluster.
"""

import os
import sys
import argparse
import tarfile
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import mimetypes


class FileClusterer:
    """Handle file clustering and archiving operations."""

    def __init__(
        self,
        input_dir,
        output_dir,
        n_clusters=5,
        compression="xz",
        xz_level=6,
        zstd_level=3,
    ):
        """
        Initialize the file clusterer.

        Args:
            input_dir: Directory containing files to cluster
            output_dir: Directory where archives will be saved
            n_clusters: Number of clusters for GMM
            compression: Compression type ('xz', 'zstd', or 'both')
            xz_level: XZ compression level (0-9, default: 6)
            zstd_level: Zstandard compression level (1-22, default: 3)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.n_clusters = n_clusters
        self.compression = compression
        self.xz_level = xz_level
        self.zstd_level = zstd_level
        self.files = []
        self.features = []
        self.clusters = None

        # Validate compression levels
        if not 0 <= xz_level <= 9:
            raise ValueError(f"XZ compression level must be 0-9, got {xz_level}")
        if not 1 <= zstd_level <= 22:
            raise ValueError(
                f"Zstandard compression level must be 1-22, got {zstd_level}"
            )

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Print compression settings
        if compression in ["xz", "both"]:
            print(f"XZ compression level: {xz_level}")
        if compression in ["zstd", "both"]:
            print(f"Zstandard compression level: {zstd_level}")
        print()

    def scan_files(self, recursive=True):
        """
        Scan the input directory for files.

        Args:
            recursive: Whether to scan subdirectories
        """
        print(f"Scanning files in {self.input_dir}...")

        if recursive:
            for root, dirs, filenames in os.walk(self.input_dir):
                for filename in filenames:
                    filepath = Path(root) / filename
                    if filepath.is_file():
                        self.files.append(filepath)
        else:
            for filepath in self.input_dir.iterdir():
                if filepath.is_file():
                    self.files.append(filepath)

        print(f"Found {len(self.files)} files")

        if len(self.files) < self.n_clusters:
            print(
                f"Warning: Number of files ({len(self.files)}) is less than "
                f"number of clusters ({self.n_clusters})"
            )
            self.n_clusters = max(1, len(self.files))

    def extract_features(self):
        """
        Extract features from files for clustering.

        Features:
        - File size (normalized)
        - File type (encoded as numeric)
        - Modification time (normalized)
        - File extension hash
        """
        print("Extracting features from files...")

        feature_list = []
        valid_files = []

        for filepath in self.files:
            try:
                stat = filepath.stat()

                # Feature 1: File size in bytes
                size = stat.st_size

                # Feature 2: File type (encoded)
                mime_type, _ = mimetypes.guess_type(str(filepath))
                if mime_type:
                    # Simple encoding: hash the mime type
                    type_encoding = hash(mime_type.split("/")[0]) % 1000
                else:
                    type_encoding = 0

                # Feature 3: Modification time (seconds since epoch)
                mtime = stat.st_mtime

                # Feature 4: Extension encoding
                ext = filepath.suffix.lower()
                ext_encoding = hash(ext) % 1000 if ext else 0

                features = [size, type_encoding, mtime, ext_encoding]
                feature_list.append(features)
                valid_files.append(filepath)

            except Exception as e:
                print(f"Warning: Could not process {filepath}: {e}")
                continue

        self.files = valid_files
        self.features = np.array(feature_list)

        print(f"Extracted features from {len(self.files)} files")

    def cluster_files(self):
        """Perform k-means clustering on the files."""
        if len(self.files) == 0:
            print("Error: No files to cluster")
            return

        print(f"Clustering files into {self.n_clusters} groups...")

        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(self.features)

        # Perform GMM clustering
        gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
        self.clusters = gmm.fit_predict(features_normalized)

        # Print cluster statistics
        print("\nCluster Statistics:")
        for i in range(self.n_clusters):
            cluster_files = [f for f, c in zip(self.files, self.clusters) if c == i]
            total_size = sum(f.stat().st_size for f in cluster_files)
            print(
                f"  Cluster {i}: {len(cluster_files)} files, "
                f"Total size: {self._format_size(total_size)}"
            )

    def create_archives(self):
        """Create compressed tar archives for each cluster."""
        if self.clusters is None:
            print("Error: Files must be clustered before archiving")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for cluster_id in range(self.n_clusters):
            cluster_files = [
                f for f, c in zip(self.files, self.clusters) if c == cluster_id
            ]

            if not cluster_files:
                continue

            print(f"\nProcessing Cluster {cluster_id} ({len(cluster_files)} files)...")

            # Create base archive name
            archive_base = self.output_dir / f"cluster_{cluster_id}_{timestamp}"

            # Create tar archive based on compression type
            if self.compression in ["xz", "both"]:
                self._create_tar_archive(cluster_files, f"{archive_base}.tar.xz", "xz")

            if self.compression in ["zstd", "both"]:
                self._create_tar_archive(
                    cluster_files, f"{archive_base}.tar.zst", "zstd"
                )

    def _create_tar_archive(self, files, archive_path, compression_type):
        """
        Create a compressed tar archive.

        Args:
            files: List of file paths to include
            archive_path: Path for the output archive
            compression_type: 'xz' or 'zstd'
        """
        print(f"  Creating archive: {Path(archive_path).name}")

        if compression_type == "xz":
            print(f"    Compression: XZ level {self.xz_level}")
        elif compression_type == "zstd":
            print(f"    Compression: Zstandard level {self.zstd_level}")

        try:
            if compression_type == "xz":
                # Use tar with xz compression and custom compression level
                # XZ preset: 0 (fast) to 9 (best compression)
                with tarfile.open(archive_path, f"w:xz", preset=self.xz_level) as tar:
                    for filepath in files:
                        # Add file with relative path
                        arcname = filepath.relative_to(self.input_dir)
                        tar.add(filepath, arcname=arcname)

            elif compression_type == "zstd":
                # Create uncompressed tar first, then compress with zstd
                temp_tar = f"{archive_path}.tmp"

                with tarfile.open(temp_tar, "w") as tar:
                    for filepath in files:
                        arcname = filepath.relative_to(self.input_dir)
                        tar.add(filepath, arcname=arcname)

                # Compress with zstd at specified level
                try:
                    import zstandard as zstd

                    with open(temp_tar, "rb") as f_in:
                        with open(archive_path, "wb") as f_out:
                            compressor = zstd.ZstdCompressor(level=self.zstd_level)
                            compressor.copy_stream(f_in, f_out)

                    # Remove temporary tar file
                    os.remove(temp_tar)

                except ImportError:
                    # Fall back to command line zstd if library not available
                    print(
                        "    Warning: zstandard library not found, using command line..."
                    )
                    result = subprocess.run(
                        ["zstd", f"-{self.zstd_level}", temp_tar, "-o", archive_path],
                        capture_output=True,
                    )

                    if result.returncode == 0:
                        os.remove(temp_tar)
                    else:
                        print(
                            f"    Error: zstd compression failed: {result.stderr.decode()}"
                        )
                        return

            # Get final archive size
            archive_size = Path(archive_path).stat().st_size
            print(f"    Archive created: {self._format_size(archive_size)}")

        except Exception as e:
            print(f"    Error creating archive: {e}")

    @staticmethod
    def _format_size(size_bytes):
        """Format bytes to human-readable size."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Cluster files using GMM and create compressed archives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i /path/to/files -o /path/to/archives -k 5 -c xz
  %(prog)s -i ./documents -o ./archives -k 10 -c both --recursive
  %(prog)s -i ./data -o ./backup -c xz --xz-level 9
  %(prog)s -i ./media -o ./compressed -c zstd --zstd-level 15

Compression Levels:
  XZ:    0 (fastest) to 9 (best compression), default: 6
  Zstd:  1 (fastest) to 22 (best compression), default: 3
         Levels 1-19 are normal, 20-22 are "ultra" compression
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory containing files to cluster",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for compressed archives"
    )
    parser.add_argument(
        "-k", "--clusters", type=int, default=5, help="Number of clusters (default: 5)"
    )
    parser.add_argument(
        "-c",
        "--compression",
        choices=["xz", "zstd", "both"],
        default="xz",
        help="Compression type (default: xz)",
    )
    parser.add_argument(
        "--xz-level",
        type=int,
        default=6,
        help="XZ compression level 0-9 (default: 6, 0=fast 9=best)",
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=3,
        help="Zstandard compression level 1-22 (default: 3, 1=fast 22=best)",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Scan subdirectories recursively"
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        sys.exit(1)

    # Create clusterer and process files
    try:
        clusterer = FileClusterer(
            input_dir=args.input,
            output_dir=args.output,
            n_clusters=args.clusters,
            compression=args.compression,
            xz_level=args.xz_level,
            zstd_level=args.zstd_level,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    clusterer.scan_files(recursive=args.recursive)

    if len(clusterer.files) == 0:
        print("No files found to process")
        sys.exit(0)

    clusterer.extract_features()
    clusterer.cluster_files()
    clusterer.create_archives()

    print(f"\n✓ Processing complete! Archives saved to {args.output}")


if __name__ == "__main__":
    main()
