# Google Colab AI Prompt: Build 3DES Key Extraction Pipeline

## 🎯 Mission Objective
Build a production-grade **3DES Key Recovery Pipeline** in Google Colab using proven research architectures. Extract KENC, KMAC, and KDEK keys from single-trace power analysis with >98% accuracy on Mastercard smartcard traces.

---

## 📋 Technical Foundation

### Core Architecture (Use These Exact Repositories)
1. **Primary: ANSSI-FR/ASCAD CNN** (Peer-reviewed, published research)
   - Repository: https://github.com/ANSSI-FR/ASCAD
   - **Why**: Proven solution for deep learning side-channel attacks, specifically designed for S-Box targeting
   - **Use**: CNN architecture, training methodology, rank-based key recovery

2. **Implementation Guide: Google SCAAML**
   - Repository: https://github.com/google/scaaml
   - **Why**: Production-grade TensorFlow pipeline structure
   - **Use**: Data loading, preprocessing pipeline, end-to-end attack workflow

3. **Preprocessing Reference: ChipWhisperer**
   - Repository: https://github.com/newaetech/chipwhisperer
   - **Why**: Best practices for trace normalization and POI selection
   - **Use**: Z-Score normalization, trace alignment techniques

4. **Optional: LeakDetectAI (for architecture optimization)**
   - Repository: https://github.com/LeakDetectAI/deep-learning-sca
   - **Why**: Automated Neural Architecture Search
   - **Use**: If you want to experiment with different CNN topologies

### 3DES Implementation Reference
- **DES S-Box Logic**: https://gist.github.com/BlackRabbit-github/2924939
- **Use**: S-Box targeting and key reconstruction logic

---

## 🔑 Key Specifications (CRITICAL - Must Match Exactly)

### 3DES Keys (Total: 48 bytes / 96 hex characters)
| Key Name | Purpose | Size (Bytes) | Size (Hex) | Example |
|:---|:---|---:|---:|:---|
| **KENC** | Key Encryption | 16 | 32 | `A1B2C3D4E5F6071829384756ABCDEF01` |
| **KMAC** | Key MAC | 16 | 32 | `1122334455667788990AABBCCDDEEFF0` |
| **KDEK** | Key Data Encryption | 16 | 32 | `FEDCBA9876543210ABCDEF0123456789` |

**Attack Target**: 8 S-Boxes per DES round (First-Round S-Box attack)
- Each S-Box: 6-bit input → 4-bit output (64 input classes, 16 output classes)
- Total: 8 S-Boxes to predict per trace

### RSA Keys (For Reference - Not in This Pipeline)
| Component | Size (Bytes) | Size (Hex) |
|:---|---:|---:|
| Modulus (N) | 256 | 512 |
| P (Prime 1) | 128 | 256 |
| Q (Prime 2) | 128 | 256 |
| DP, DQ, QINV | 128 each | 256 each |

---

## 📊 Dataset Information

### Training Data (Mastercard Only)
**Location**: Will be uploaded to Colab  
**Files**:
- `traces_data_1000T_1.npz` (1,000 traces, ~1 GB)
- `traces_data_2000T_2.npz` (2,000 traces, ~2 GB)
- `traces_data_2000T_3.npz` (2,000 traces, ~2 GB)
- `traces_data_2000T_4.npz` (2,000 traces, ~2 GB)
- `traces_data_3000T_5.npz` (3,000 traces, ~3 GB)

**Total**: 10,000 Mastercard 3DES traces (~10 GB)  
**Channels**: C1 (Power), C3 (Clock) - **NO I/O channel in this dataset**  
**Reference Keys**: Available in NPZ files for validation

> [!IMPORTANT]
> **Critical Data Gap**: There is NO Visa 3DES training data. All 10,000 traces are Mastercard-only.

> [!WARNING]
> **Training Philosophy - Learning vs Memorization**
> 
> **Goal**: Build models that **learn the approach to reach ground truth**, not memorize specific card signatures.
> 
> **Critical Requirements**:
> - ✅ Train ONLY on Mastercard 3DES traces (10,000 traces)
> - ❌ **DO NOT use test card data** for cross-validation during training
> - ❌ **DO NOT use template CSV files** for reference
> - ⚠️ **Reason**: Including test cards causes memorization instead of learning
> 
> **What This Means**:
> - Models must learn **general SCA attack methodology** (S-Box targeting, power analysis patterns)
> - Should NOT learn "this specific trace pattern = this specific key"
> - Use proper train/validation splits from the 10,000 Mastercard traces only
> - Validation accuracy reflects generalization to **unseen Mastercard traces**

---

## 🏗️ Pipeline Architecture: 4 Operational Modes

### Mode 1: TRAIN
**Purpose**: Train the ASCAD CNN ensemble for 3DES S-Box prediction.

**Implementation Steps**:
1. **Data Loading**:
   - Load all 5 NPZ files (10,000 traces total)
   - Extract power traces from C1 channel
   - Extract reference 3DES keys (KENC, KMAC, KDEK) from labels
   - Split: 80% train, 20% validation (from Mastercard data only)

2. **Preprocessing** (Use ChipWhisperer techniques):
   - **Z-Score normalization** per trace: `(trace - mean) / (std + 1e-8)`
   - **POI Selection**: Identify Points of Interest (optional, can use full trace)
   - **Trace Alignment**: Ensure all traces are synchronized

3. **Label Generation**:
   - Generate S-Box targets for First-Round DES attack
   - Target: 8 S-Box outputs (one per S-Box)
   - Each S-Box output: 4-bit value (0-15 classes)

4. **Model Architecture** (Adapt from ANSSI-FR/ASCAD):
   - **Decoupled Ensemble**: Train 8 separate CNN models (one per S-Box)
   - **Why**: Eliminates bit collisions between S-Box predictions
   - **CNN Structure** (per S-Box model):
     ```
     Input: Power trace (normalized)
     Conv1D(64 filters, kernel=11, activation=relu)
     BatchNormalization()
     AveragePooling1D(pool_size=2)
     Conv1D(128 filters, kernel=11, activation=relu)
     BatchNormalization()
     AveragePooling1D(pool_size=2)
     Flatten()
     Dense(256, activation=relu, kernel_regularizer=l2(1e-2))
     Dropout(0.5)
     Dense(16, activation=softmax)  # 16 classes for 4-bit S-Box output
     ```

5. **Training Configuration**:
   - **Optimizer**: Adam (lr=0.001)
   - **Loss**: Categorical Crossentropy
   - **Regularization**: L2 (weight_decay=1e-2) to prevent overfitting
   - **Epochs**: 100-200 (use early stopping)
   - **Batch Size**: 128
   - **Metrics**: Accuracy, Top-5 Accuracy

6. **Output**:
   - Save 8 trained models: `sbox_0.h5`, `sbox_1.h5`, ..., `sbox_7.h5`
   - Save normalization parameters (mean, std) for each trace
   - Save training history and validation metrics

---

### Mode 2: ATTACK
**Purpose**: Recover 3DES keys from blind traces using trained models.

**Implementation Steps**:
1. **Load Blind Trace**:
   - Accept single trace or batch of traces (NPZ format)
   - Extract C1 (Power) channel

2. **Preprocessing**:
   - Apply same Z-Score normalization as training
   - Use saved normalization parameters

3. **Ensemble Prediction**:
   - Run all 8 S-Box models on the trace
   - For each S-Box, get probability distribution over 16 classes
   - Output: 8 probability vectors (one per S-Box)

4. **Rank-Based Key Recovery**:
   - For each S-Box, rank all possible 6-bit inputs by likelihood
   - Combine S-Box predictions to reconstruct 48-byte key material
   - Use DES key schedule to derive KENC, KMAC, KDEK

5. **Key Reconstruction**:
   - Extract bytes 0-15: KENC
   - Extract bytes 16-31: KMAC
   - Extract bytes 32-47: KDEK
   - Convert to **Clean Hex format** (no spaces, no prefixes)

6. **Output**:
   - JSON with recovered keys and confidence scores:
     ```json
     {
       "3DES_KENC": "A1B2C3D4E5F6071829384756ABCDEF01",
       "3DES_KMAC": "1122334455667788990AABBCCDDEEFF0",
       "3DES_KDEK": "FEDCBA9876543210ABCDEF0123456789",
       "confidence_scores": [0.98, 0.95, 0.97, ...],
       "rank_0_success": true
     }
     ```

---

### Mode 3: VALIDATE
**Purpose**: Verify model accuracy against reference keys.

**Implementation Steps**:
1. **Load Test Set**:
   - Use 20% validation split from training data
   - Ensure test traces have known reference keys

2. **Run Attack Mode**:
   - Apply attack pipeline to all test traces
   - Compare predicted keys with reference keys

3. **Calculate Metrics**:
   - **Rank 0 Success Rate**: % of traces with exact key match
   - **Top-5 Rank**: % of traces with correct key in top 5 predictions
   - **Per-S-Box Accuracy**: Accuracy breakdown for each of 8 S-Boxes
   - **Confidence Distribution**: Histogram of prediction confidence scores

4. **Identify Failing S-Boxes**:
   - If any S-Box has <90% accuracy, flag for retraining
   - Generate per-S-Box confusion matrices

5. **Output**:
   - Validation report (markdown format):
     ```markdown
     # Validation Report
     - **Rank 0 Success**: 98.5%
     - **Top-5 Rank**: 99.8%
     - **S-Box Accuracies**: [99%, 98%, 97%, 96%, 98%, 99%, 97%, 98%]
     - **Failing S-Boxes**: None
     ```

---

### Mode 4: OUTPUT
**Purpose**: Generate client-ready CSV/Excel files.

**Implementation Steps**:
1. **Parse Recovered Keys**:
   - Load attack results (JSON from Mode 2)
   - Extract KENC, KMAC, KDEK

2. **Format Output Columns**:
   | Column | Type | Format | Example |
   |:---|:---|:---|:---|
   | `Card_Type` | String | Mastercard/Visa | `Mastercard` |
   | `PAN` | String | 16 digits | `5412345678901234` |
   | `Expiry` | String | MMYY | `1228` |
   | `ATC` | String | **Decimal digits** | `0001` (NOT `0x0001`) |
   | `AIP` | String | Clean Hex | `1234` |
   | `IAD` | String | Clean Hex | `ABCDEF0123456789` |
   | `3DES_KENC` | String | Clean Hex (32 chars) | `A1B2C3D4E5F6071829384756ABCDEF01` |
   | `3DES_KMAC` | String | Clean Hex (32 chars) | `1122334455667788990AABBCCDDEEFF0` |
   | `3DES_KDEK` | String | Clean Hex (32 chars) | `FEDCBA9876543210ABCDEF0123456789` |

3. **Excel Safety** (CRITICAL):
   - Prefix all hex strings with `'` to force text mode
   - Example: `'A1B2C3D4E5F6071829384756ABCDEF01`
   - **Why**: Prevents Excel from converting to scientific notation

4. **Default Value Injection**:
   - If AIP/IAD missing: Use default based on card type
   - Mastercard default AIP: `1980`
   - Mastercard default IAD: `06010A03A00000`

5. **Generate Dual Outputs**:
   - **`output_clean.csv`**: No quotes, for EMV tools
   - **`output_excel_safe.csv`**: With `'` prefix, for Excel viewing

6. **Output**:
   - Two CSV files ready for client delivery

---

## 🔧 Critical Implementation Details

### 1. Preprocessing (Z-Score Normalization)
```python
import numpy as np

def normalize_trace(trace):
    """Per-trace Z-Score normalization for generalization"""
    mean = np.mean(trace)
    std = np.std(trace)
    return (trace - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
```

### 2. Decoupled S-Box Architecture
```python
# DO NOT share weights between S-Box models
# Train 8 separate models to avoid bit collisions

models = []
for sbox_idx in range(8):
    model = build_cnn_model(input_shape=(trace_length, 1), num_classes=16)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    models.append(model)
```

### 3. Key Reconstruction Logic
```python
def reconstruct_3des_keys(sbox_predictions):
    """
    Reconstruct KENC, KMAC, KDEK from 8 S-Box predictions
    
    Args:
        sbox_predictions: List of 8 probability distributions (each 16 classes)
    
    Returns:
        dict: {'KENC': str, 'KMAC': str, 'KDEK': str}
    """
    # Reverse DES S-Box operation to recover subkeys
    # Combine subkeys to reconstruct full 48-byte key material
    # Split into KENC (0-15), KMAC (16-31), KDEK (32-47)
    
    # Implementation details:
    # 1. For each S-Box, get top prediction (argmax)
    # 2. Reverse S-Box lookup to get 6-bit input
    # 3. Combine 8 S-Box inputs to get round key
    # 4. Use DES key schedule to derive master key
    # 5. Split master key into KENC, KMAC, KDEK
    
    pass  # Implement based on DES gist reference
```

### 4. Known Issues to Avoid
- ❌ **RSA QINV Bug**: Not applicable to 3DES pipeline
- ❌ **Key Duplication**: Ensure KENC, KMAC, KDEK are independently extracted
- ❌ **Excel Scientific Notation**: Use `'` prefix for all hex strings
- ❌ **Overfitting to Templates**: Use L2 regularization and Z-Score normalization
- ❌ **Shared Weights**: Train 8 separate models, not one multi-output model

---

## 📦 Expected Deliverables

1. **Colab Notebook** with 4 executable modes (train, attack, validate, output)
2. **Trained Models** (8 S-Box models: `sbox_0.h5` to `sbox_7.h5`, ~100 MB total)
3. **Validation Report** showing Rank 0 accuracy on test set
4. **Sample Output CSV** matching client template format
5. **README** with usage instructions for each mode

---

## ✅ Success Criteria

- ✅ **Mastercard Rank 0**: 100% key recovery on Mastercard test traces
- ✅ **Generalization**: >90% accuracy on unseen Mastercard traces
- ✅ **Output Format**: CSV passes client template validation
- ✅ **Reproducibility**: Pipeline runs end-to-end in fresh Colab environment
- ✅ **No Memorization**: Models learn attack methodology, not specific traces

---

## 🚀 Execution Instructions

### Step 1: Environment Setup
```python
# Install dependencies
!pip install numpy scipy tensorflow scikit-learn pandas openpyxl

# Clone repositories
!git clone https://github.com/ANSSI-FR/ASCAD.git
!git clone https://github.com/google/scaaml.git

# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
```

### Step 2: Data Upload
Upload ONLY the 5 Mastercard training NPZ files to Colab:
- `traces_data_1000T_1.npz`
- `traces_data_2000T_2.npz`
- `traces_data_2000T_3.npz`
- `traces_data_2000T_4.npz`
- `traces_data_3000T_5.npz`

**DO NOT upload**:
- ❌ Test card data
- ❌ Template CSV files
- ❌ Blind test traces

### Step 3: Load and Preprocess Data
```python
# Load all NPZ files
traces_list = []
labels_list = []

for npz_file in ['traces_data_1000T_1.npz', 'traces_data_2000T_2.npz', ...]:
    data = np.load(npz_file)
    traces_list.append(data['traces'])  # Assuming 'traces' key
    labels_list.append(data['labels'])  # Assuming 'labels' key

# Concatenate all traces
all_traces = np.concatenate(traces_list, axis=0)  # Shape: (10000, trace_length)
all_labels = np.concatenate(labels_list, axis=0)  # Shape: (10000, key_length)

# Normalize traces
normalized_traces = np.array([normalize_trace(t) for t in all_traces])

# Train/validation split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    normalized_traces, all_labels, test_size=0.2, random_state=42
)
```

### Step 4: Build CNN Models (Adapt from ASCAD)
```python
def build_cnn_model(input_shape, num_classes=16):
    """Build CNN for S-Box prediction (adapted from ASCAD)"""
    model = keras.Sequential([
        keras.layers.Conv1D(64, 11, activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling1D(2),
        keras.layers.Conv1D(128, 11, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu', 
                          kernel_regularizer=keras.regularizers.l2(1e-2)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

### Step 5: Train Models (One per S-Box)
```python
# Train 8 separate models
models = []
for sbox_idx in range(8):
    print(f"Training S-Box {sbox_idx}...")
    
    # Generate S-Box labels for this S-Box
    y_train_sbox = generate_sbox_labels(y_train, sbox_idx)  # Implement this
    y_val_sbox = generate_sbox_labels(y_val, sbox_idx)
    
    # Build and compile model
    model = build_cnn_model(input_shape=(X_train.shape[1], 1))
    model.compile(optimizer='adam', loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Train
    history = model.fit(
        X_train, y_train_sbox,
        validation_data=(X_val, y_val_sbox),
        epochs=100,
        batch_size=128,
        callbacks=[keras.callbacks.EarlyStopping(patience=10)]
    )
    
    # Save model
    model.save(f'sbox_{sbox_idx}.h5')
    models.append(model)
```

### Step 6: Validate
```python
# Mode 3: Validate
rank_0_count = 0
for i in range(len(X_val)):
    predicted_key = attack_single_trace(X_val[i], models)  # Implement this
    reference_key = y_val[i]
    
    if predicted_key == reference_key:
        rank_0_count += 1

rank_0_accuracy = rank_0_count / len(X_val)
print(f"Rank 0 Accuracy: {rank_0_accuracy * 100:.2f}%")
```

### Step 7: Attack Blind Traces
```python
# Mode 2: Attack
def attack_single_trace(trace, models):
    """Recover 3DES keys from a single trace"""
    # Normalize trace
    normalized = normalize_trace(trace)
    
    # Predict S-Box outputs
    sbox_predictions = []
    for model in models:
        pred = model.predict(normalized.reshape(1, -1, 1))
        sbox_predictions.append(pred[0])
    
    # Reconstruct keys
    keys = reconstruct_3des_keys(sbox_predictions)
    return keys
```

### Step 8: Generate Output
```python
# Mode 4: Output
results = []
for trace in blind_traces:
    keys = attack_single_trace(trace, models)
    results.append({
        'Card_Type': 'Mastercard',
        'PAN': '5412345678901234',  # Extract from metadata
        'Expiry': '1228',
        'ATC': '0001',  # Decimal format
        'AIP': '1980',
        'IAD': '06010A03A00000',
        '3DES_KENC': keys['KENC'],
        '3DES_KMAC': keys['KMAC'],
        '3DES_KDEK': keys['KDEK']
    })

# Save to CSV
df = pd.DataFrame(results)

# Clean CSV (no quotes)
df.to_csv('output_clean.csv', index=False, quoting=0)

# Excel-safe CSV (with ' prefix)
df_excel = df.copy()
for col in ['3DES_KENC', '3DES_KMAC', '3DES_KDEK', 'AIP', 'IAD']:
    df_excel[col] = "'" + df_excel[col]
df_excel.to_csv('output_excel_safe.csv', index=False, quoting=0)
```

---

## 🎓 Development Strategy: Phased Approach

> [!IMPORTANT]
> **Two-Stage Development Plan**
> 
> At the **initial stage**, I want **working and accurate models separately** for both:
> 1. **3DES Pipeline** (this prompt)
> 2. **RSA Pipeline** (separate prompt/notebook)
> 
> Each pipeline should be:
> - ✅ Fully functional and independently testable
> - ✅ Achieving >98% accuracy on its respective task
> - ✅ Generating correct output formats
> 
> At the **finalization stage**, I will **merge these 2 pipelines** to create a single unified pipeline that:
> - Handles both 3DES and RSA key recovery
> - Automatically detects card type and applies the appropriate models
> - Generates a single consolidated CSV output with all recovered keys
> 
> **For this Colab notebook**: Focus exclusively on the **3DES pipeline**. The RSA integration will come later.

---

*Prompt Generated: February 12, 2026*
