param(
    [ValidateSet('train','evaluate')][string]$Task = 'train',
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$Seed = 7,
    [ValidateSet('on','off')][string]$Shield = 'on',
    [ValidateSet('validation','test')][string]$Split = 'validation',
    [string[]]$SACModels = @(),
    [string[]]$ResidualModels = @(),
    [switch]$Resume,
    [switch]$Smoke,
    [int]$StopAfter = 0
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$enhancedTmp = Join-Path $PSScriptRoot 'tmp'
New-Item -ItemType Directory -Force -Path $enhancedTmp | Out-Null
$env:TEMP = $enhancedTmp
$env:TMP = $enhancedTmp
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:MPLCONFIGDIR = Join-Path $enhancedTmp 'matplotlib'
$env:NUMBA_CACHE_DIR = Join-Path $enhancedTmp 'numba'
$env:TORCH_HOME = Join-Path $enhancedTmp 'torch'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $enhancedTmp 'torchinductor'
$env:XDG_CACHE_HOME = Join-Path $enhancedTmp 'cache'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$enhancedPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $enhancedPython)) { throw 'Run .\run.ps1 -Task setup first.' }
$enhancedArgs = @('-m','gridguard.enhanced_experiment',$Task,'--output',$Output,'--seed',"$Seed",'--shield',$Shield,'--split',$Split)
if ($Resume) { $enhancedArgs += '--resume' }
if ($Smoke) { $enhancedArgs += '--smoke' }
if ($StopAfter -gt 0) { $enhancedArgs += @('--stop-after',"$StopAfter") }
foreach ($model in $SACModels) { $enhancedArgs += @('--sac-model',$model) }
foreach ($model in $ResidualModels) { $enhancedArgs += @('--residual-model',$model) }
& $enhancedPython @enhancedArgs
if ($LASTEXITCODE -ne 0) { throw "Enhanced task failed with exit code $LASTEXITCODE." }
