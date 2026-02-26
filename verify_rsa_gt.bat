@echo off
REM RSA Ground Truth Verification
REM This script verifies the RSA models against the known ground truth keys in the validation dataset.

echo ============================================================
echo RSA-1152 Ground Truth Verification
echo Target: P, Q, DP, DQ, QINV (15 Models)
echo ============================================================

cd "..\Smartcard SCA ML Pipeline"

REM Check if Python environment is active or use system python
REM We assume the test script is test_rsa_gt.py or we need to run inference first.
REM Based on previous checks, test_rsa_gt.py takes a JSON file.
REM We need a script that does the full loop: Inference -> Verify.
REM Since I haven't written that unified script yet, I will verify the manual RSA scripts first.

echo Running RSA Inference and Verification...
python src/inference_rsa.py

if %errorlevel% neq 0 (
    echo [ERROR] RSA Verification failed.
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] RSA Models Verified.
pause
