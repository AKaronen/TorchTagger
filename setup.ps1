# To run before using the codes (PowerShell)
$cwd = (Get-Location).Path
if ([string]::IsNullOrEmpty($env:PYTHONPATH)) {
    $env:PYTHONPATH = $cwd
} else {
    $env:PYTHONPATH = "$env:PYTHONPATH$([System.IO.Path]::PathSeparator)$cwd"
}

$env:CI_COMMIT_REF_NAME = 'local'