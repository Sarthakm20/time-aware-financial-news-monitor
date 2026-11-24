# Daily live-feed update: fetch + score new headlines, check alerts, and (if a Gemini key is set) run the
# live accuracy check. Logs go to data\live\logs\<date>.log.
#
# Run by hand:      powershell -ExecutionPolicy Bypass -File scripts\live_update.ps1
# Schedule daily:   see README, "Live news feed".
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$logDir = Join-Path $root "data\live\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

"=== $(Get-Date -Format o) live-update alerts" | Out-File -Append -Encoding utf8 $log
& $py -m src.pipeline live-update alerts *>&1 | Out-File -Append -Encoding utf8 $log

$envFile = Join-Path $root ".env"
$hasKey = $env:GEMINI_API_KEY -or ((Test-Path $envFile) -and (Select-String -Path $envFile -Pattern '^\s*GEMINI_API_KEY\s*=\s*\S' -Quiet))
if ($hasKey) {
    "=== $(Get-Date -Format o) live-annotate" | Out-File -Append -Encoding utf8 $log
    & $py -m src.pipeline live-annotate *>&1 | Out-File -Append -Encoding utf8 $log
}
