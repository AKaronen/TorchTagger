# To run before using the codes (PowerShell)
# Dot-source this script so env vars/functions persist in current shell:
#   . .\setup.ps1

Set-StrictMode -Version Latest

function Add-UniquePath {
    param(
        [string]$Path,
        [System.Collections.Generic.HashSet[string]]$Seen,
        [System.Collections.Generic.List[string]]$Out
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return
    }

    try {
        $resolved = (Resolve-Path -Path $Path -ErrorAction Stop).Path
    }
    catch {
        $resolved = $Path
    }

    if ($Seen.Add($resolved)) {
        $Out.Add($resolved)
    }
}

$repoRoot = if ($PSScriptRoot) {
    (Resolve-Path $PSScriptRoot).Path
}
else {
    (Get-Location).Path
}
$taggerRoot = Join-Path $repoRoot "tagger"

$pathsToAdd = [System.Collections.Generic.List[string]]::new()
$pathsToAdd.Add($repoRoot)

if (Test-Path $taggerRoot) {
    $pathsToAdd.Add($taggerRoot)
    Get-ChildItem -Path $taggerRoot -Directory -Recurse | ForEach-Object {
        $pathsToAdd.Add($_.FullName)
    }
}

$separator = [System.IO.Path]::PathSeparator
$existingPaths = @()
if (-not [string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $existingPaths = $env:PYTHONPATH -split [Regex]::Escape($separator)
}

$seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
$merged = [System.Collections.Generic.List[string]]::new()

foreach ($path in $pathsToAdd) {
    Add-UniquePath -Path $path -Seen $seen -Out $merged
}
foreach ($path in $existingPaths) {
    Add-UniquePath -Path $path -Seen $seen -Out $merged
}

$env:PYTHONPATH = ($merged -join $separator)
$env:CI_COMMIT_REF_NAME = "local"

function Invoke-TaggerModule {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string]$Module,
        [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )

    python -m $Module @Args
}

Write-Host "Configured PYTHONPATH for repo:" -ForegroundColor Green
Write-Host "  $repoRoot" -ForegroundColor Green
Write-Host "Entries in PYTHONPATH: $($merged.Count)" -ForegroundColor Green
Write-Host "Example: Invoke-TaggerModule tagger.data.parsers.make_data tagger/data/data_config_template.yaml" -ForegroundColor Green