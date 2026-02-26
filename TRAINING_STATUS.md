# Training Status & Next Steps

## ✅ Memory Issue Fixed

**Problem**: Training failed when trying to load all 10,000 traces (required ~5GB+ RAM)

**Solution**: Limited dataset to first 4 NPZ files (7,000 traces total):
- `traces_data_1000T_1.npz` (1,000 traces)
- `traces_data_2000T_2.npz` (2,000 traces)
- `traces_data_2000T_3.npz` (2,000 traces)
- `traces_data_2000T_4.npz` (2,000 traces)

**Skipped**: `traces_data_3000T_5.npz` (3,000 traces) - causes memory allocation error

---

## 🚀 Training Command

```bash
python main.py --mode train --epochs 50 --batch-size 64
```

**Training on**: 7,000 traces (5,600 train / 1,400 validation)

**Expected time**: 30-60 minutes on CPU

---

## 📊 What to Expect

### During Training

You'll see output like:

```
============================================================
STEP 1: Data Preparation
============================================================
Loading traces...
✓ Total traces loaded: 7000

Normalizing traces (Z-Score per trace)...
✓ Normalized 7000 traces

✓ Train/Val split:
  Training: 5600 traces (80%)
  Validation: 1400 traces (20%)

============================================================
STEP 2: Label Generation
============================================================
Generating S-Box labels for KENC...
✓ Generated labels shape: (7000, 8)

============================================================
STEP 3: Model Building
============================================================
Building S-Box 0 model...
Building S-Box 1 model...
...

============================================================
STEP 4: Training
============================================================
Training S-Box 0 Model
Epoch 1/50
88/88 [==============================] - 45s - loss: 2.7xxx - accuracy: 0.xxxx - val_loss: 2.6xxx - val_accuracy: 0.xxxx
...
```

### After Training Completes

You'll find:
- **Models**: `models/sbox_0.h5` through `models/sbox_7.h5`
- **Training history**: `results/training_history.png`
- **Metadata**: `results/training_metadata.json`

---

## ⏭️ After Training: Next Steps

### 1. Validate Models

```bash
python main.py --mode validate
```

**Expected output**:
- `results/validation_report.md` - Detailed metrics
- `results/validation_metrics.json` - JSON metrics

**Target metrics**:
- Rank 0 Success Rate: ≥ 99%
- All S-Box accuracies: ≥ 90%

### 2. Attack Test Traces

```bash
# Single trace
python main.py --mode attack --input Input/Mastercard/traces_data_1000T_1.npz --trace-index 0

# All traces in file
python main.py --mode attack --input Input/Mastercard/traces_data_1000T_1.npz
```

### 3. Generate Output CSV

```bash
python main.py --mode output --input results/attack_results.json
```

**Output files**:
- `results/output_clean.csv` (for EMV tools)
- `results/output_excel_safe.csv` (for Excel)

---

## 🔧 If Training Fails

### Low Accuracy (< 90%)

**Possible causes**:
- Not enough training epochs
- Too much regularization
- Model too simple

**Solutions**:
- Increase epochs: `--epochs 100`
- Reduce L2 regularization in `model_builder.py`
- Train longer

### Memory Errors

**Solution**: Already implemented - using only 7,000 traces

### Training Too Slow

**Solutions**:
- Reduce batch size: `--batch-size 32`
- Reduce epochs for testing: `--epochs 20`
- Use GPU if available

---

## 📈 Expected Results

With 7,000 training traces and strong regularization:

**Optimistic scenario**:
- Validation accuracy: 95-99%
- Rank 0 success: 90-95%

**Realistic scenario**:
- Validation accuracy: 85-95%
- Rank 0 success: 70-85%

**If accuracy is low**:
- The model is learning SCA patterns but may need more data or tuning
- Consider reducing regularization or increasing model capacity

---

## 💡 Key Points

1. **Dataset limitation**: Training on 7,000 traces (not 10,000) due to memory constraints
2. **Generalization focus**: Strong L2 regularization (1e-2) and 50% dropout prevent overfitting
3. **Z-Score normalization**: Ensures model learns patterns, not absolute values
4. **Decoupled models**: 8 separate S-Box models avoid bit collisions

---

## Current Status

✅ **Memory issue resolved**
✅ **Training started**
⏳ **Waiting for training to complete** (~30-60 min)
⏳ **Validation pending**
⏳ **Output generation pending**
