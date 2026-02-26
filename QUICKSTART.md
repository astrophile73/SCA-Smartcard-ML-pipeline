# Quickstart (Headless)

All commands are headless and work on Linux or Windows (paths differ).

## 1) Preprocess (Extract Features + Save POIs)

```bash
python main.py --mode preprocess --input_dir Input1 --processed_dir Processed --opt_dir Optimization --scan_type all
```

## 2) Train (3DES)

```bash
python main.py --mode train --input_dir Input1 --processed_dir Processed --opt_dir Optimization --scan_type 3des --epochs 100
```

## 3) Attack (3DES and/or RSA)

```bash
python main.py --mode attack --input_dir Input --processed_dir Processed_Attack --output_dir results --scan_type all --opt_dir Optimization
```

Output reports:
- `results/Final_Report_<card_type>_<target_key>.csv`
- `results/Final_Report_<card_type>_<target_key>.xlsx` (if `openpyxl` is installed)

