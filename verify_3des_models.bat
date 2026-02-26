@echo off
REM 3DES Model Verification Wrapper
REM This script runs a single-trace attack to verify the 24 trained models.

echo ============================================================
echo 3DES Model Verification
echo Target: KENC, KMAC, KDEK (24 Models)
echo Input: Input/Mastercard/traces_data_1000T_1.npz (Trace #0)
echo ============================================================

python main.py --mode attack --input "Input/Mastercard/traces_data_1000T_1.npz" --trace-index 0

if %errorlevel% neq 0 (
    echo [ERROR] 3DES Attack failed. Please check logs.
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] 3DES Key Recovery Verified.
pause
