# Project Summary & Collaboration Guide

This document is a simplified summary of everything we have discovered, fixed, and set up for the SCA Smartcard Pipeline so far. It is designed to get you and your collaborator on exactly the same page immediately.

---

## 1. What Are We Trying to Do?
We are building a **Side-Channel Analysis (SCA) Machine Learning Pipeline** to recover cryptographic keys from EMV smartcards using power traces (C1), clock alignment (C3), and I/O data (C7).

**The Targets:**
- **3DES Keys:** Extract the 16-byte session keys (KENC, KMAC, KDEK).
- **RSA Keys:** Extract the CRT private key components (P, Q, DP, DQ, QINV) from the `VERIFY` command.
- **PIN:** Decrypt the user PIN from the `VERIFY` command.

---

## 2. Current State of the Pipeline
- ✅ **RSA Key Recovery:** Working. The byte-level ensemble models successfully predict P and Q, and then use math (`rsatool` logic) to derive DP, DQ, and QINV.
- ❌ **3DES Key Recovery:** Fails to recover the correct keys (produces garbage/incorrect hex strings).
- ❌ **PIN Extraction:** Completely missing from the codebase. The output Excel shows a `PIN` column, but it is always empty.

---

## 3. Why are the 3DES Keys Wrong? (The 4 Major Bugs)
We dug deep into the codebase and logs and found exactly why 3DES recovery is failing. Every one of these needs to be fixed:

1. **🔴 The "Variance Fallback" Bug (CRITICAL):**
   In `src/feature_eng.py`, the code tries to find Points of Interest (POIs) by correlating the power traces with expected S-Box outputs. **This correlation fails on every single run.** Because it fails, the code falls back to picking POIs based on signal variance (noise). The models are currently being trained on noise instead of cryptographic leakage.
2. **🔴 The Visa Poison Data Bug (CRITICAL):**
   In `label_generator.py` and `src/gen_labels.py`, if a trace is missing a 3DES key (which happens for all Visa traces), the code silently asserts a hardcoded Green Visa key (`2315208C9110AD...`) to generate training labels. This poisons the Mastercard training data with fake Visa labels, destroying model accuracy.
3. **🟠 The Stage-2 Cascading Failure:**
   The 3DES attack is 2 stages. Stage 2 requires the recovered key from Stage 1 (K1) to compute an intermediate ciphertext. If K1 is wrong, the ciphertext is wrong, making K2 recovery impossible. There is no confidence check stopping a bad K1 from ruining K2.
4. **🟠 The 8 Missing Bits in Inverse PC2:**
   When reconstructing the 64-bit key from the 48-bit round key, `src/crypto.py` simply sets the 8 discarded bits to `0` and picks that single guess. It should enumerate all 256 possibilities and verify which one decrypts the ciphertext.

---

## 4. Compliance with Client Specifications
We audited the pipeline against the two client Word docs (`Track2-3DES-From-4Channel Data.docx` and `RSA-Pin-4Chaneel Data.docx`).

**What we did right:**
- We extract the Track 2 equivalent data (PAN, ATC) from the C7 I/O line using APDU parsing (`src/ingest.py`).
- We use the correct triggers (`GENERATE AC` for 3DES, `VERIFY` for RSA).
- We use ML-based CPA (Correlation Power Analysis via ZaidNet/ASCAD models), which perfectly satisfies the spec's CPA requirement.

**What we missed (Needs to be built):**
- **PIN Decryption:** The RSA spec dictates we must parse the encrypted PIN block from the C7 data, do an RSA decryption (`m = c^d mod n`), and parse the resulting ISO 9564 hex to extract the actual PIN digits. This code doesn't exist yet.
- **3DES Validation:** The 3DES spec dictates we must validate our recovered key by recalculating the MAC. We currently don't do this.

---

## 5. What We Inherited (External Code)
To save you time figuring out where the math came from, here are the external libraries the pipeline relies on:
- **`pyDes` (by Todd Whiteman / BlackRabbit):** The entire foundation of our 3DES pipeline. We use its S-Box tables, key schedule generation, and permutation tables (PC1, PC2).
- **ZaidNet CNN:** The primary model architecture used for 3DES (from a 2020 TCHES paper).
- **ASCAD CNN:** A legacy model architecture still in the codebase but not the primary one.
- **PyCryptodome & rsatool:** Used for all the RSA integer math and CRT component derivation.

---

## 6. GitHub & Data Setup
We cleaned up the massive repository and pushed it to GitHub: `https://github.com/astrophile73/SCA-Smartcard-ML-pipeline`

**Important Note on Repo Size:**
The project folder was **270 GB** on disk, but the actual source code was only **1 MB**.
- We pushed only the code, `.docx`/`.pdf` reports, and markdown notes.
- We deliberately **excluded the 53 GB of `.keras` and `.pth` models** via `.gitignore`. The models are currently broken anyway, and we will need to retrain them from scratch once we fix the POI and Visa Poisoning bugs.
- We also excluded the 20 GB of raw `Input1/` `.npz` traces and added empty placeholder folders (`Input/`, `Output/`, `Processed/`, `Optimization/`) with `.gitkeep` files so the pipeline doesn't crash on a fresh clone.

**What's Next?**
Share the 20 GB `Input1/` folder with your collaborator via Google Drive. Have them clone the GitHub repo, drop the `Input1/` folder inside, and then you can both start tackling the bugs above!
