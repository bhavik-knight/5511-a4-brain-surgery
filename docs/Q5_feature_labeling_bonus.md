# Q5 - Feature Labeling and Cluster Validation

Primary implementation files:

- `src/brain_surgery/clustering.py`
- `scripts/verify_pilot.py`

## Theoretical Goal

Group latent features into semantic families and test whether those groups align
with metadata categories instead of arbitrary geometric artifacts.

## Implementation

- Applies Spherical K-Means by L2-normalizing feature vectors before clustering.
- Uses a dynamic elbow heuristic (rate-of-change slowdown in SSE) to choose $k$.
- Performs final K-Means assignment at selected $k$.
- Computes category-purity statistics per cluster using metadata category votes.

## Results

- Produces `cluster_report.json` in each run directory with:
  - selected cluster count,
  - per-cluster dominant category,
  - cluster purity,
  - theme tags (`Tactical`, `Historical`, `Clubs`),
  - cross-cluster theme summary.
- This turns feature labeling from subjective inspection into a metadata-driven
  validation protocol.
