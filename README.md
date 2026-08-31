# MycoGraphX

Self-contained Streamlit application for serving the `mico_EEGNN` model.
The tool accepts a single SMILES structure or a CSV file with up to 200
structures, applies calibrated thresholds, and provides downloadable results.
For individual structures, it also generates target-specific counterfactual
explanation maps. Every target prediction includes an epistemic uncertainty
score estimated with Monte Carlo dropout.

## Contents

- `app.py`: Streamlit Community Cloud entrypoint.
- `mycographx/`: application package, model code, rules, and inference logic.
- `mycographx/artifacts/`: checkpoint, hyperparameters, thresholds, and
  applicability-domain data used at runtime.
- `scripts/`: maintenance utilities for regenerating derived artifacts.
- `tests/`: executable smoke and Streamlit interface tests.
- `.github/workflows/ci.yml`: GitHub Actions test workflow.
- `Dockerfile`: reproducible production container.
- `packages.txt`: native libraries required by RDKit on Streamlit Cloud.

## Repository layout

```text
.
|-- .github/workflows/ci.yml
|-- .streamlit/config.toml
|-- mycographx/
|   |-- artifacts/
|   |-- applicability.py
|   |-- augmentations.py
|   |-- explainability.py
|   |-- inference.py
|   |-- model.py
|   |-- rules.py
|   `-- utils.py
|-- scripts/build_applicability_artifacts.py
|-- tests/
|-- app.py
|-- Dockerfile
|-- packages.txt
|-- requirements.txt
`-- requirements-dev.txt
```

## Run locally

Python 3.11 is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Run with Docker

```powershell
docker build -t mycographx .
docker run --rm -p 8501:8501 mycographx
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository with this directory as its root.
2. Commit all files, including `mycographx/artifacts/*.pth` and `*.npz`.
3. Push the repository to GitHub and confirm that the `CI` workflow passes.
4. Select the repository and the root `app.py` entrypoint in Streamlit
   Community Cloud.
5. Use Python 3.11 when runtime configuration is available.

The `.pth` file is approximately 4.5 MB and can be versioned directly.
`requirements.txt` installs the CPU distribution of PyTorch to reduce deployment
size. Before publishing, add the citation, dataset version, and validation
metrics required for scientific use to the Model tab.

## Tests

Install the runtime dependencies and execute the four test programs from the
repository root:

```powershell
python tests/smoke_test.py
python tests/streamlit_startup_test.py
python tests/activity_profile_test.py
python tests/streamlit_aas_test.py
```

GitHub Actions runs the same checks for every push and pull request.

## Batch input

The CSV must contain a `SMILES` column. An `ID`, `compound_id`, `name`, or
`molecule_id` column is optional. Common delimiters are detected automatically.

## Threshold-adjusted activity score

The activity profile displays its main progress bar as a threshold-adjusted
activity score for easier comparison across endpoints with different
classification thresholds. The piecewise-linear display transformation maps
zero to 0, the endpoint-specific threshold to 0.5, and one to 1. Consequently,
scores below 0.5 are inactive and scores at or above 0.5 are active for every
endpoint. The original probability and threshold remain visible as numeric
reference columns. This adjusted score is also included in CSV downloads as
`threshold_adjusted_score__<target>`.

The adjusted score is a threshold-relative visualization index, not a
recalibrated probability. Epistemic uncertainty remains on the original
probability scale.

The embedding KDE applicability-domain result is presented in its dedicated
section and is intentionally omitted from the activity-profile table.

## Epistemic uncertainty

The application keeps the model in evaluation mode while temporarily enabling
only its dropout layers. It runs 30 Monte Carlo passes by default and reports,
for every target, the mean predicted probability and the population standard
deviation of those probabilities as the epistemic uncertainty score. The EEGNN
graph generator is also resampled between passes. The number of passes can be
adjusted from 2 to 50 in the interface; higher values give a more stable estimate
but increase processing time. CSV downloads include one
`epistemic_uncertainty__<target>` column per target.

## Gaussian-KDE embedding applicability domain

Every prediction is assessed separately for each endpoint using the mean
Euclidean distance to the five nearest training molecules carrying a label for
that endpoint. One final embedding is extracted for every training molecule
immediately before the prediction layer of the selected checkpoint. Embeddings
are standardized with training-only statistics and L2-normalized before the
distance calculation. The resulting leave-one-out mean 5-NN distances form a
one-dimensional coverage score for each endpoint.

A reflection-corrected Gaussian kernel density estimate (KDE) is fitted to every
endpoint's complete training-score distribution using Scott's bandwidth rule.
The query status is determined from the KDE cumulative probability: `Inside` at
or below 95%, `Borderline` between 95% and 99%, and `Outside` above 99%. Thus,
KDE participates directly in the domain decision while the underlying distances
still use all 364 embedding dimensions. The validation split audits coverage;
the test split is never used for calibration.

The UI shows the leave-one-out distance histogram, its Gaussian KDE curve, the
95% and 99% KDE limits, and the query position. It also shows the five nearest
labeled embedding references. No dimensionality-reduction projection is used.

The deployable artifacts are `mycographx/artifacts/embedding_domain.npz` and
`mycographx/artifacts/embedding_domain_metadata.json`. Regenerate them from the
original train and validation tables with explicit input paths:

```powershell
pip install -r requirements-dev.txt
python scripts/build_applicability_artifacts.py --train path/to/train.csv --validation path/to/val.csv
```

The artifact builder verifies task coverage and records hashes of the model and
data sources. KDE-domain status, mean embedding-neighbor distance, KDE CDF
percentile, and KDE density are included in CSV downloads. Domain status is a
coverage diagnostic, not a guarantee of correctness or biological activity.

## Counterfactual maps

After running a single prediction, select a target and generate the map. The
rulebook applies every valid local chemical transformation to every atom in the
molecule and measures the resulting paired changes in the model score. There is
no per-atom sampling limit. Consequently, map generation can take substantially
longer for large structures or structures admitting many transformations:

- red: the original region raises the score relative to the counterfactuals;
- blue: the original region lowers the score relative to the counterfactuals;
- intensity: relative effect magnitude within the analyzed compound.

Two map scales are available:

- `Relative contrast` is the default and follows the centering step used by the
  original explanation pipeline. Red and blue identify higher and lower atom
  effects within the current molecule; they do not indicate effect direction.
- `Absolute direction` preserves the sign of the counterfactual effect. Red
  means that the original region raises the score, while blue means that it
  lowers the score.

The colors do not directly represent the Active/Inactive class. That class is
determined by comparing the model score with the calibrated target threshold,
both shown beside the map. A fully red absolute-direction map may occur for an
inactive molecule when every tested transformation lowers a score that was
already below the threshold.

The map and the complete list of evaluated transformations can be downloaded.
The per-atom aggregation is used internally to color the map but is not shown as
a redundant result table. These effects describe the model's local sensitivity
and do not establish biological causality.

## Disclaimer

This tool is intended for research and experimental prioritization. It must not
be used for diagnosis, clinical decision-making, or confirmation of biological
activity.

## License

This repository follows the HolisticGNN project license: Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). See `LICENSE` for
the applicable terms.
