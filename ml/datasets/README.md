# Dataset annotations

The NDJSON files preserve the exported image dimensions, annotations and
train/validation/test assignments. Image filenames and source identifiers have
been replaced with deterministic sample aliases. Access signatures and account
links have been removed.

Source images are not included. The `media.example.invalid` and
`datasets.example.invalid` URLs are deliberate placeholders and cannot be used
for downloads. Obtain authorized source images before running dataset-based
training or accuracy evaluation. Local unit tests use separate fixtures and
stubs.

## Evaluation boundaries

- `monkey-cocotree-v2.ndjson` remains the default dataset manifest. The files in
  `archive/` retain their original record counts and split assignments.
- Existing filename grouping prefixes are preserved. The current v2 export
  contains no benchmark origin prefixes, so `--benchmark` cannot recover the
  older per-source groups from this manifest. Use a separately identified
  benchmark export when evaluating an individual source group.
- Media URL keys are aliases, not original CDN content hashes. They are not
  suitable for direct comparison with a live Platform dataset or an old image
  cache. Restore an authorized dataset export before using the Platform drift
  checker; do not interpret alias mismatches as annotation drift.
- `tests/f1_baseline.json` and `tests/f1_baseline_val.json` retain historical
  reference results without changing their scores or measurement settings.
  They are not a new evaluation of this submission snapshot or a fresh accuracy
  measurement of the supplied checkpoint.
- `tests/extra_test_cases.ndjson` currently contains no active per-image
  assertions. Aggregate scores do not guarantee retention of every individual
  detection capability.

The model checkpoint remains usable for inference. Its numerical tensor data
is unchanged; identifying training-path metadata has been replaced. The model
contains the `monkey` and `human` classes.
