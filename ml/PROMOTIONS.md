# Model and evaluation notes

## Included checkpoint

`backend/best.pt` is a YOLO26n detector with two classes: `monkey=0` and
`human=1`. Monitoring requests `classes=[0]` before tracking, saving records and
sending notifications. The human class is a classification alternative; it does
not introduce human tracking, counts or alerts. A human misclassified as monkey
can still produce a false alert.

The latest training round added human hard negatives. The checkpoint was
accepted with a test-gate exception, described below. Both stored baseline files
remain unchanged so subsequent candidates still face the stronger reference.

## Recorded evaluation

These results were measured on the original frozen benchmark image bytes and
labels. They are historical evaluation results, not a fresh run supplied with
this checkout. Source field images and access credentials are not included.

| Frozen benchmark | Images / monkey boxes | Previous TP / FP / FN | Included TP / FP / FN | Previous F1 | Included F1 | Gate |
|---|---:|---:|---:|---:|---:|---|
| Test | 342 / 699 | 555 / 71 / 144 | 533 / 63 / 166 | 0.837736 | 0.823166 | FAIL |
| Validation | 179 / 363 | 311 / 35 / 52 | 312 / 27 / 51 | 0.877292 | 0.888889 | PASS |

Evaluation used native resolution with a minimum long side of 1600, gray-world
correction only above cast ratio 1.5, MPS FP16, confidence 0.5 and matching IoU
0.5. Test recall changed from 0.7940 to 0.7625: 22 fewer matched boxes, not 22
fewer distinct animals.

The stored test baseline F1 is 0.8354. With tolerance 0.01, the passing floor is
**0.8254**; the included model misses it by **0.002234**. This was an accepted
deployment exception, not a passing gate. The result does not establish that
adding a human class inherently reduces macaque recall.

Four time-separated clips from one camera/day contained 458 stored raw frames.
The previous model produced 180 monkey boxes on these frames; the included
model produced zero. Selected snapshots were visually reviewed as people or
background. Full-clip ground truth and a continuous operating-time denominator
were not established. These are trigger-selected frame predictions, not alert
counts, an independent cross-day test, or evidence of macaque recall at that
camera.

## Training basis

The training dataset contained 2,300 images: 1,757 train, 195 validation and
348 test. It retained 2,282 source images, labels and splits, adding 18 training
images from 18 clips and 42 reviewed human boxes. Four added images were
background negatives. Human annotations were weak labels rather than
independently adjudicated ground truth. Ingestion reencoded 2,237 inherited
images; the frozen benchmark scores above used the original cached bytes.

Training used stock YOLO26n, image size 1920, batch 8, seed 0, AdamW learning
rate 0.001, a 1,000-epoch limit and patience 100. The checkpoint's training
history recorded 273 epochs, with the best checkpoint at epoch 173.

## Evaluating another candidate

`tests/eval_ratchet.py` checks both test and validation splits. Its basis check
rejects comparisons with incompatible image-size policy, confidence, precision,
split or dataset size. `tests/extra_test_cases.ndjson` supports additional
presence checks for difficult examples, but currently contains no active assertions.

Provide authorized local benchmark images and a matching NDJSON file before
running an accuracy evaluation. Keep the frozen evaluation splits separate
from training data, and preserve the stronger baselines when a model is accepted
as an exception. A score from a different dataset, device or preprocessing
configuration is a different measurement and should be reported separately.

The frozen test and selected replay clips have already influenced model
selection. They are regression/challenge sets, not an untouched final exam.
Broader claims require independent grouped evaluation and field measurements
of false alerts and missed sightings.
