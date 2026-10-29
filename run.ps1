param(
    [ValidateSet('setup', 'info', 'check', 'simulate', 'baselines', 'env-check', 'sac-smoke', 'sac-evaluate', 'sac-formal', 'demo', 'test')]
    [string]$Task = 'check',
    [string]$Config = 'configs/ieee33.toml',
    [string]$BaselineConfig = 'configs/baselines.toml',
    [string]$EnvironmentConfig = 'configs/environment.toml',
    [string]$SACConfig = 'configs/sac.toml',
    [string]$Model = '',
    [ValidateSet('validation', 'test')]
    [string]$Split = 'validation',
    [string]$Output = '',
    [string]$IndexUrl = '',
    [string]$TorchWheel = '',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
# All settings are limited to this process and its children.
$taskTmp = Join-Path $PSScriptRoot 'tmp'
New-Item -ItemType Directory -Force -Path $taskTmp | Out-Null
$env:TEMP = $taskTmp
$env:TMP = $taskTmp
$env:PIP_CACHE_DIR = Join-Path $taskTmp 'pip'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = Join-Path $taskTmp 'matplotlib'
$env:NUMBA_CACHE_DIR = Join-Path $taskTmp 'numba'
$env:TORCH_HOME = Join-Path $taskTmp 'torch'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $taskTmp 'torchinductor'
$env:XDG_CACHE_HOME = Join-Path $taskTmp 'cache'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if ($Task -eq 'setup') {
    if (-not (Test-Path -LiteralPath $taskPython)) {
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 venv creation failed.' }
    }
    # Install the explicit CPU build first; regular dependency resolution follows.
    & $taskPython -c "import importlib.metadata as m; import importlib.util as u; import sys; sys.exit(0 if u.find_spec('torch') is not None and m.version('torch') == '2.8.0+cpu' else 1)"
    if ($LASTEXITCODE -ne 0) {
        if ($TorchWheel) {
            if (-not (Test-Path -LiteralPath $TorchWheel -PathType Leaf)) { throw 'Specified PyTorch wheel not found.' }
            & $taskPython -m pip install --no-compile --no-deps --timeout 120 --retries 5 $TorchWheel
        } else {
            & $taskPython -m pip install --no-compile --no-deps --timeout 120 --retries 5 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
        }
        if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
        & $taskPython -c "import importlib.metadata as m; import sys; sys.exit(0 if m.version('torch') == '2.8.0+cpu' else 1)"
        if ($LASTEXITCODE -ne 0) { throw 'Expected torch==2.8.0+cpu.' }
    }
    $installArgs = @('-m', 'pip', 'install', '--no-compile', '--timeout', '120', '--retries', '5', '-e', '.[dev]', '-c', 'requirements-lock.txt')
    if ($IndexUrl) { $installArgs += @('--index-url', $IndexUrl) }
    & $taskPython @installArgs
} elseif (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Run .\run.ps1 -Task setup first (requires Python 3.12).'
} elseif ($Task -eq 'test') {
    & $taskPython -m pytest
} else {
    $taskArgs = @('-m', 'gridguard', $Task, '--config', $Config)
    if ($Task -eq 'sac-formal' -and -not $PSBoundParameters.ContainsKey('SACConfig')) { $SACConfig = 'configs/sac_formal.toml' }
    if ($Task -in @('baselines', 'env-check', 'sac-smoke', 'sac-evaluate', 'sac-formal', 'demo')) { $taskArgs += @('--baseline-config', $BaselineConfig) }
    if ($Task -in @('env-check', 'sac-smoke', 'sac-evaluate', 'sac-formal', 'demo')) { $taskArgs += @('--environment-config', $EnvironmentConfig) }
    if ($Task -in @('sac-smoke', 'sac-evaluate', 'sac-formal', 'demo')) {
        if ($Task -eq 'demo' -and -not $PSBoundParameters.ContainsKey('SACConfig')) { $SACConfig = 'configs/sac_formal.toml' }
        $taskArgs += @('--sac-config', $SACConfig)
    }
    if ($Resume) { $taskArgs += @('--resume') }
    if ($Task -eq 'sac-evaluate') {
        if (-not $Model) { throw 'sac-evaluate requires -Model <checkpoint.zip>.' }
        $taskArgs += @('--model', $Model, '--split', $Split)
    }
    if ($Task -eq 'demo' -and $Model) { $taskArgs += @('--model', $Model) }
    if ($Output) { $taskArgs += @('--output', $Output) }
    & $taskPython @taskArgs
}
if ($LASTEXITCODE -ne 0) { throw "Task '$Task' failed with exit code $LASTEXITCODE." }
