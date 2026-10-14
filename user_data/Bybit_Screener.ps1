# Bybit_Screener.ps1
# Launches the Screener freqtrade strategy using the Screener_config.json
# config file. Activates the project virtual environment (.venv) first.
#
# Usage:  .\Bybit_Screener.ps1 [extra freqtrade args...]

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$VenvDir     = Join-Path $ProjectRoot '.venv'
$Activate    = Join-Path $VenvDir 'Scripts\Activate.ps1'
$Config      = Join-Path $PSScriptRoot 'Screener_config.json'

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

$env:FREQTRADE__EXCHANGE__NAME = 'bybit'
$env:FREQTRADE__EXCHANGE__KEY = $env:BYBIT_DEMO_API_KEY
$env:FREQTRADE__EXCHANGE__SECRET = $env:BYBIT_DEMO_API_SECRET
# Use Bybit futures market for perpetual contracts (USDT-margined)
$env:FREQTRADE__EXCHANGE__MARKET_TYPE = 'futures'

# Force UTF-8 console output: the 'smartmoneyconcepts' package prints a unicode
# banner at import that crashes under the default Windows cp1252 encoding.
$env:PYTHONIOENCODING = 'utf-8'

# Pin the working directory to the freqtrade project root so user_data, the
# relative sqlite db_url and log paths resolve consistently regardless of
# where the script is launched from.
Set-Location -LiteralPath $ProjectRoot

# Run freqtrade using the activated environment; fall back to the venv executable
# if activation was blocked (e.g. by execution policy).

# Optional: scoreboard_sort_order can be passed as first positional argument (volume|ratio)
$ScoreboardSortOrder = "ratio"
$ExtraArgs = @()
foreach ($a in $args) {
    $ExtraArgs += $a
}

# Simple positional override: first arg is sort order if it matches allowed values
if ($ExtraArgs.Count -ge 1) {
    $first = $ExtraArgs[0].ToString().ToLowerInvariant()
    if ($first -eq "volume" -or $first -eq "ratio") {
        $ScoreboardSortOrder = $first
        $ExtraArgs = $ExtraArgs[1..($ExtraArgs.Count - 1)]
    }
}

# Export to environment so the strategy can read it
$env:FVG_SCOREBOARD_SORT_ORDER = $ScoreboardSortOrder

if (Get-Command freqtrade -ErrorAction SilentlyContinue) {
    freqtrade trade -c $Config --strategy Screener @ExtraArgs
} else {
    & (Join-Path $VenvDir 'Scripts\freqtrade.exe') trade -c $Config --strategy Screener @ExtraArgs
}

exit $LASTEXITCODE