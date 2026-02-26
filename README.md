# Smartcard SCA ML Pipeline (3DES + RSA)

Reproducible, headless pipeline for recovering cryptographic material (3DES session keys and RSA CRT components)
from ChipWhisperer Husky traces.

This repo contains legacy experiments, but the supported entrypoint is:

- `main.py` + `src/` (PyTorch-based, "pure ML" attack path)

## Install

```bash
pip install -r requirements.txt
```

## Data Expectations

- Input: a directory containing one or more `traces_data_*.npz` files.
- The directory may contain:
  - only 3DES traces, or
  - only RSA traces, or
  - both (attack auto-detects which are present).

## Output Format (Client Template)

Reports are generated as CSV and XLSX with *exactly* these column headers:

`PROFILE, TRACK2, AIP, IAD, 3DES_KENC, 3DES_KMAC, 3DES_KDEK, PIN, RSA_CRT_P, RSA_CRT_Q, RSA_CRT_DP, RSA_CRT_DQ, RSA_CRT_QINV`

No extra columns are added.

## Usage

### 1) Preprocess / Feature Extraction

```bash
python main.py --mode preprocess --input_dir Input1 --processed_dir Processed --opt_dir Optimization
```

Notes:
- POIs and the alignment reference trace are stored under:
  - `Optimization/pois_3des/` and `Optimization/pois_rsa/`
- Processed features are stored under:
  - `Processed/3des/` and `Processed/rsa/`

### 2) Train (3DES)

```bash
python main.py --mode train --input_dir Input1 --processed_dir Processed --opt_dir Optimization --scan_type 3des --epochs 100
```

RSA training is skipped unless dataset-internal RSA labels exist.

### 3) Attack (3DES and/or RSA)

```bash
python main.py --mode attack --input_dir Input --processed_dir Processed_Attack --output_dir results --scan_type all --opt_dir Optimization
```

Attack mode forces POI reuse (`--use_existing_pois`) to avoid recomputing POIs without secrets.

## Pure-ML Guarantee (What This Pipeline Does NOT Do)

- Does not read external spreadsheets/JSON to fill in keys.
- Does not hardcode "known" GreenVisa keys/masks.
- Does not output any ground-truth secrets into the final report.

