# Superseded dataset exports

These six exports precede `monkey-cocotree-v2`. They retain their original image
record counts, annotations and split assignments, with identifying filenames
and source URLs replaced. Source images are not included. See
[dataset access and evaluation boundaries](../README.md).

The evaluation script searches only the parent `ml/datasets/` directory by
default. Pass an archive manifest explicitly with `--ndjson` to inspect or
evaluate that dataset once authorized source images are available. Its results
need a baseline measured on the same dataset and settings.

Earlier merged exports used filename prefixes to identify source benchmarks.
The current v2 export does not carry those prefixes, so `--benchmark` cannot
recover the old groups from this manifest. Pre-merge measurements are not
directly interchangeable with v2 measurements: upload re-encoding and dataset
membership changes can affect results.
