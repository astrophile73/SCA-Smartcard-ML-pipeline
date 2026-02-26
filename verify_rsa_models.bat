@echo off
REM RSA Model Verification Wrapper
REM This script runs the pre-configured RSA test in the sibling pipeline.

echo ============================================================
echo RSA-1152 Model Verification
echo Target: Gold Standard Ensemble (P, Q, DP, DQ, QINV)
echo ============================================================

cd "..\Smartcard SCA ML Pipeline"

REM Check if Python environment is active or use system python
python src/inference_rsa.py

if %errorlevel% neq 0 (
    echo [ERROR] RSA Inference failed. Please check logs.
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] RSA Logic Verified.
pause
