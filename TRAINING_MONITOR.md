# Training Monitor Guide (UPDATED)

**Training Started**: 2026-02-13 11:05 AM IST  
**Current Progress**: 7 / 24 models complete (29%)  
**Last Check**: 2026-02-13 05:06 PM IST  
**Estimated Completion**: 2026-02-14 07:30 AM IST (Tomorrow morning)

---

## 📊 Detailed Progress

| Milestone | Status | Completed At |
|:----------|:-------|:-------------|
| **Data Preparation** | ✅ Done | 11:08 AM |
| **KENC S-Box 0** | ✅ Done | 12:00 PM |
| **KENC S-Box 1** | ✅ Done | 12:50 PM |
| **KENC S-Box 2** | ✅ Done | 01:42 PM |
| **KENC S-Box 3** | ✅ Done | 02:35 PM |
| **KENC S-Box 4** | ✅ Done | 03:26 PM |
| **KENC S-Box 5** | ✅ Done | 04:18 PM |
| **KENC S-Box 6** | ✅ Done | 05:06 PM |
| **KENC S-Box 7** | ⏳ Training | ~06:00 PM |
| **KMAC (8 Models)** | ⏳ Pending | ~01:00 AM |
| **KDEK (8 Models)** | ⏳ Pending | ~07:30 AM |

---

## ⏱️ Updated Statistics

- **Total Time Elapsed**: 6 hours
- **Avg Time Per Model**: ~51 minutes (includes early stopping and loading)
- **Total Estimated Time**: ~20-21 hours total
- **Remaining Models**: 17 models

---

## 🔍 Verification Details

The following models are confirmed saved in the `models/` directory:
- `sbox_0_kenc.h5`
- `sbox_1_kenc.h5`
- `sbox_2_kenc.h5`
- `sbox_3_kenc.h5`
- `sbox_4_kenc.h5`
- `sbox_5_kenc.h5`
- `sbox_6_kenc.h5`

Each model is approximately 492 MB, totaling ~3.4 GB of storage so far.

---

## 🎯 Next Steps After Completion

Once the final model (`sbox_7_kdek.h5`) is saved:
1. Review the generated `results/training_history_24models.png` for accuracy across all keys.
2. I will update `attack.py` and `key_recovery.py` to handle the multi-key recovery logic.
3. We will run the final validation.


---

## ✅ Success Indicators

1. **No errors** in terminal output
2. **Models appearing** in `models/` directory
3. **Validation accuracy** > 95% per model
4. **Early stopping** around epoch 20-30

---

### Issue 1: Why did training stop initially?
**Reason**: **Memory Accumulation**. In the previous version, TensorFlow was keeping every trained model (8 S-Boxes) in RAM simultaneously. Since each model is ~500MB, the system eventually ran out of memory and froze.

**Fix Applied (Ready for Resume)**:
- ✅ **One-by-One Training**: The script now builds, trains, and saves one model at a time.
- ✅ **Session Clearing**: After each model, it now calls `tf.keras.backend.clear_session()` to flush the RAM.
- ✅ **Garbage Collection**: Forces Python to release memory immediately.
- ✅ **Auto-Resume**: You can now restart anytime without losing work.

### Issue 2: TensorFlow Warnings
**Symptom**: `oneDNN` warnings  
**Solution**: Ignore - these are performance hints, not errors

### Issue 3: Slow Progress
**Symptom**: Taking longer than expected  
**Solution**: Normal - proper DES is more complex than simplified

---

## 📁 Expected Output Files

After training completes:

```
models/
├── sbox_0_kenc.h5 ... sbox_7_kenc.h5  (8 files, ~1.25 GB)
├── sbox_0_kmac.h5 ... sbox_7_kmac.h5  (8 files, ~1.25 GB)
├── sbox_0_kdek.h5 ... sbox_7_kdek.h5  (8 files, ~1.25 GB)
└── poi_indices.npy                     (1 file, ~40 KB)

results/
├── training_metadata.json              (training stats)
└── training_history_24models.png       (accuracy plots)
```

**Total size**: ~3.75 GB

---

## 🎯 Next Steps After Training

Once training completes:

1. **Check validation accuracy** in `training_metadata.json`
2. **View training plots** in `training_history_24models.png`
3. **Update attack pipeline** to use 24 models
4. **Run validation mode** to verify accuracy
5. **Test attack mode** on sample traces

---

## 💡 Tips

- **Let it run overnight** - don't interrupt
- **Check progress occasionally** - but don't worry if it seems slow
- **Monitor disk space** - ensure ~4 GB free
- **Keep terminal open** - closing it will stop training

Training is running in the background. Check back in a few hours!
