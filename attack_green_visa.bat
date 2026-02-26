@echo off
REM GreenVisa 3DES Attack Script
REM Runs the trained models on the GreenVisa dataset.
REM Note: This dataset does not contain reference keys, so we can only Output predictions.

echo ============================================================
echo GreenVisa 3DES Attack
echo Target: KENC, KMAC, KDEK (Blind Attack)
echo Input Dir: I:\freelance\Smartcard SCA ML Pipeline\GreenVisa\GreenVisa
echo ============================================================

cd "i:\freelance\SCA-Smartcard-Pipeline-3"

REM Run attack (auto-detect 3DES traces in the directory)
python main.py --mode attack --input_dir "I:\freelance\Smartcard SCA ML Pipeline\GreenVisa\GreenVisa" --processed_dir "Processed_GreenVisa" --output_dir "results_green_visa" --scan_type 3des --card_type visa --opt_dir "Optimization" --use_existing_pois

if %errorlevel% neq 0 (
    echo [ERROR] Attack failed.
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Attack Complete. Please check results_green_visa for the output report.
pause
