@echo off
REM Build the SNP-specialised Ball Covariance kernel (Windows / MinGW).
REM   cd cball_ext
REM   compile_snp_win.bat
REM
REM NOTE: the output name is deliberately NOT cball_ext.dll -- that file is the
REM existing Ball BDD kernel built by compile_win.bat, and overwriting it would
REM break bcov._bdd_matrix_bias_c.
setlocal

set CC=gcc
if not "%1"=="" set CC=%1

%CC% -O3 -shared -o csnpbcov.dll snp_bcov.c -lm
if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)
echo built csnpbcov.dll
endlocal
