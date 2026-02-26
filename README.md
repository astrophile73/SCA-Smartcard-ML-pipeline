# SCA Smartcard ML Pipeline (3DES + RSA)

ML-powered Side-Channel Analysis pipeline for recovering **3DES session keys** (KENC, KMAC, KDEK) and **RSA CRT private key** components (P, Q, DP, DQ, QINV) from EMV smartcard power traces.

> **Status:** RSA recovery works. **3DES key recovery is producing incorrect keys** — see [Known Issues](#known-issues) below.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Setup](#setup)
- [Getting the Data & Models](#getting-the-data--models)
- [Architecture](#architecture)
- [Usage](#usage)
- [Known Issues (Why 3DES Keys Are Wrong)](#known-issues)
- [Recommended Fix Sequence](#recommended-fix-sequence)
- [References](#references)

---

## Project Overview

This pipeline takes raw smartcard power traces (captured from 4 channels: C1=Power, C3=Clock, C7=I/O, Trigger) and extracts cryptographic keys using ML-based Correlation Power Analysis.

| Target | Algorithm | Method | Status |
|---|---|---|---|
| 3DES KENC/KMAC/KDEK | 2-key TDES (16 bytes) | ZaidNet CNN → S-Box CPA → 2-stage key recovery | ❌ Broken |
| RSA P, Q, DP, DQ, QINV | RSA-CRT (512-bit primes) | Multi-head byte classifier → rsatool derivation | ✅ Working |
| PIN | ISO 9564 | RSA decrypt of VERIFY command | ❌ Not implemented |

**Input:** `.npz` trace files + `.csv` metadata from ChipWhisperer Husky  
**Output:** CSV/XLSX with one row per card containing extracted keys

---

## Setup

### 1. Clone and Install

```bash
git clone https://github.com/astrophile73/SCA-Smartcard-ML-pipeline.git
cd SCA-Smartcard-ML-pipeline
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch, NumPy, Pandas, SciPy, tqdm, openpyxl

### 2. Directory Structure

After cloning, you'll have these empty folders (data goes here):

```
Input/          ← Place trace NPZ files here
Output/         ← Pipeline writes results here
Processed/      ← Feature extraction saves here (auto-generated)
Optimization/   ← POIs, normalization stats (auto-generated)
```

---

## Getting the Data & Models

The trace data (~20 GB) and pre-trained models (~7 GB) are **not in this repo** (too large for GitHub).

| What | Size | Download |
|---|---|---|
| `Input1/` (Mastercard + Visa traces) | ~20 GB | [Google Drive link — TBD] |
| `models/` (pre-trained, broken models) | ~7 GB | [Google Drive link — TBD] |

After downloading:
```
# Extract into repo root
SCA-Smartcard-ML-pipeline/
├── Input1/
│   ├── Mastercard/     ← 3DES traces (.npz + .csv)
│   └── Visa/           ← RSA traces (.npz)
└── models/
    ├── Ensemble_ZaidNet/   ← 3DES models (.pth)
    └── rsa/                ← RSA models (.pth)
```

> **Note:** The existing models produce wrong 3DES keys. They're included so you can run the pipeline end-to-end and observe the failures before fixing.

---

## Architecture

### Key Files

| File | What It Does |
|---|---|
| `main.py` | Pipeline entry point — orchestrates preprocess → train → attack |
| `src/ingest.py` | Loads NPZ trace files, filters by card type (Mastercard/Visa) |
| `src/feature_eng.py` | **POI selection** (correlation-based) + feature extraction |
| `src/preprocess.py` | Trace alignment via FFT cross-correlation |
| `src/gen_labels.py` | Computes S-Box output labels for training |
| `src/train_ensemble.py` | Trains ZaidNet ensemble (5 models × 8 S-Boxes × 2 stages) |
| `src/inference_3des.py` | 2-stage 3DES key recovery (K1 via RK1 → K2 via RK16) |
| `src/inference_rsa.py` | RSA CRT component recovery |
| `src/crypto.py` | DES tables (S-Box, PC1, PC2, IP, E), key reconstruction, EMV derivation |
| `src/model_zaid.py` | ZaidNet CNN architecture (Zaid et al. TCHES 2020) |
| `src/pyDes.py` | Pure-Python DES engine |

### Pipeline Flow

```
NPZ traces → ingest.py (filter) → feature_eng.py (POI + extract)
    → gen_labels.py (S-Box labels) → train_ensemble.py (train ZaidNet)
    → inference_3des.py (2-stage attack) → output_gen.py (CSV/XLSX)
```

**For the full architecture with diagrams, see:** [`pipeline_technical_report.md`](pipeline_technical_report.md)

---

## Usage

### Preprocess (Extract Features + POIs)

```bash
python main.py --mode preprocess --input_dir Input1 --processed_dir Processed --opt_dir Optimization --scan_type all
```

### Train (3DES Models)

```bash
python main.py --mode train --input_dir Input1 --processed_dir Processed --opt_dir Optimization --scan_type 3des --epochs 100
```

### Attack (Run Inference)

```bash
python main.py --mode attack --input_dir Input --processed_dir Processed --output_dir Output --scan_type all --opt_dir Optimization
```

### Full Pipeline (All Steps)

```bash
python main.py --mode full --input_dir Input1 --processed_dir Processed --output_dir Output --opt_dir Optimization --scan_type all --epochs 100
```

---

## Known Issues

> **Read the full bug analysis:** [`pipeline_technical_report.md`](pipeline_technical_report.md) — Section 7

### 🔴 Bug 1: POI Correlation Always Fails (CRITICAL)

**File:** `src/feature_eng.py`

The correlation-based POI search **fails on every run** and falls back to variance-based POIs. This means models learn noise instead of actual DES leakage.

```
feature_eng - WARNING - Correlation POI union failed. Falling back to Variance-based POIs.
```

### 🔴 Bug 2: Visa Data Poisons Training Labels (CRITICAL)

**Files:** `label_generator.py`, `src/ingest.py`

When Visa traces (which have no 3DES keys) leak into training, the code defaults to a hardcoded key `2315208C9110AD40`, silently generating wrong labels.

### 🟠 Bug 3: Stage 2 Cascading Failure

**File:** `src/inference_3des.py`

If Stage 1 recovers K1 incorrectly, Stage 2 uses the wrong K1 to compute intermediate ciphertext, making K2 recovery impossible.

### 🟠 Bug 4: Inverse PC2 Missing 8 Bits

**File:** `src/crypto.py`

`reconstruct_key_from_rk1()` sets 8 unknown bits to 0 instead of trying all 256 candidates. The function `generate_key_candidates_from_rk1()` exists but isn't used during inference.

---

## Recommended Fix Sequence

```
1. Fix POI correlation (feature_eng.py)       ← Makes models learn real leakage
2. Remove Visa poison (ingest.py, gen_labels)  ← Clean training labels
3. Retrain models with clean data              ← New models from scratch
4. Fix inverse PC2 (enumerate 256 candidates)  ← Correct key reconstruction
5. Add K1 confidence check before Stage 2      ← Prevent cascading failure
6. End-to-end verification                     ← Validate recovered keys
```

---

## Output Format

CSV/XLSX with these columns:

```
PROFILE, TRACK2, AIP, IAD, 3DES_KENC, 3DES_KMAC, 3DES_KDEK, PIN,
RSA_CRT_P, RSA_CRT_Q, RSA_CRT_DP, RSA_CRT_DQ, RSA_CRT_QINV
```

---

## References

| Reference | Used For |
|---|---|
| Zaid et al., TCHES 2020 | ZaidNet CNN architecture |
| NIST FIPS 46-3 | DES S-Box tables, permutation tables |
| EMV Book 2 | Session key derivation, APDU parsing |
| Kocher et al., CRYPTO 1999 | Differential Power Analysis theory |
| ISO/IEC 7816-4 | APDU command format |

---

## Documentation

| Document | Contents |
|---|---|
| [`pipeline_technical_report.md`](pipeline_technical_report.md) | Full architecture, flows, and bug analysis |
| [`Track2-3DES-From-4Channel Data.docx`](Track2-3DES-From-4Channel%20Data.docx) | Client spec: 3DES extraction |
| [`RSA-Pin-4Chaneel Data.docx`](RSA-Pin-4Chaneel%20Data.docx) | Client spec: RSA/PIN extraction |
| [`QUICKSTART.md`](QUICKSTART.md) | Minimal 3-command quickstart |
