# SMC_Notifier.ps1
# Launches the SMC_Notifier freqtrade strategy using the SMC_Notifier_config.json
# config file. Activates the project virtual environment (.venv) first.
#
# Usage:  .\SMC_Notifier.ps1 [extra freqtrade args...]

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$VenvDir     = Join-Path $ProjectRoot '.venv'
$Activate    = Join-Path $VenvDir 'Scripts\Activate.ps1'
$Config      = Join-Path $PSScriptRoot 'SMC_Notifier_config.json'

if (-not (Test-Path -LiteralPath $Activate)) {
    Write-Error "Virtual environment not found at '$VenvDir' (expected Scripts\Activate.ps1)."
    exit 1
}
if (-not (Test-Path -LiteralPath $Config)) {
    Write-Error "Config file not found at '$Config'."
    exit 1
}

# Activate the virtual environment
& $Activate

# Force UTF-8 console output: the 'smartmoneyconcepts' package prints a unicode
# banner at import that crashes under the default Windows cp1252 encoding.
$env:PYTHONIOENCODING = 'utf-8'

# Pin the working directory to the freqtrade project root so user_data, the
# relative sqlite db_url and log paths resolve consistently regardless of
# where the script is launched from.
Set-Location -LiteralPath $ProjectRoot

# Run freqtrade using the activated environment; fall back to the venv executable
# if activation was blocked (e.g. by execution policy).
if (Get-Command freqtrade -ErrorAction SilentlyContinue) {
    freqtrade trade -c $Config --strategy SMC_Notifier @args
} else {
    & (Join-Path $VenvDir 'Scripts\freqtrade.exe') trade -c $Config --strategy SMC_Notifier @args
}

exit $LASTEXITCODE