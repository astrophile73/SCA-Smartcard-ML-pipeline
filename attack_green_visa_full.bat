@echo off
REM Unified GreenVisa Attack (3DES + RSA) - NEW PIPELINE (main.py)
echo ============================================================
echo Unified GreenVisa Attack Pipeline (Pure ML)
echo Target: 3DES (KENC, KMAC, KDEK) + RSA (P, Q, DP, DQ, QINV) if present
echo Input Dir: I:\freelance\Smartcard SCA ML Pipeline\GreenVisa\GreenVisa
echo ============================================================

cd "i:\freelance\SCA-Smartcard-Pipeline-3"

REM Run the pipeline (attack auto-detects 3DES/RSA files in the input dir)
python main.py --mode attack --input_dir "I:\freelance\Smartcard SCA ML Pipeline\GreenVisa\GreenVisa" --processed_dir "Processed_GreenVisa" --output_dir "results_green_visa" --scan_type all --card_type visa --opt_dir "Optimization" --use_existing_pois

if %errorlevel% neq 0 (
    echo [ERROR] Pipeline failed.
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Report generated in results_green_visa
pause
