$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$benchmarkTemp = Join-Path $PSScriptRoot '.benchmark-temp'
New-Item -ItemType Directory -Force -Path $benchmarkTemp | Out-Null
$env:TEMP = $benchmarkTemp
$env:TMP = $benchmarkTemp

# Install every benchmark dependency explicitly in the benchmark environment.
python -m venv --without-pip --system-site-packages .benchmark-venv
if ($LASTEXITCODE -ne 0) { throw 'Could not create benchmark environment.' }
$env:PIP_NO_INDEX = '0'
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.package-cache'
& .\.benchmark-venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple --timeout 15 --retries 0 -r benchmark-requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Benchmark dependency installation failed. Check pip output.' }
& .\.benchmark-venv\Scripts\python.exe -c "import rapidfuzz, indra_ocr, paddleocr, paddle, pypdfium2, PIL"
if ($LASTEXITCODE -ne 0) { throw 'Benchmark dependency check failed. See the import error above.' }

winget install --id tesseract-ocr.tesseract --version 5.5.3 -e --accept-package-agreements --accept-source-agreements
if ($LASTEXITCODE -ne 0) { throw 'Tesseract installation failed. Check winget output.' }

Write-Host 'Installed. Run: .\.benchmark-venv\Scripts\python.exe benchmark.py'
