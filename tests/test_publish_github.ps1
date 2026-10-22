$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$fixture = Join-Path $projectRoot ('eval/publish-test-' + [guid]::NewGuid().ToString('N'))
$previousLocation = Get-Location
$global:PublishTestCalls = New-Object 'System.Collections.Generic.List[string]'
$global:PublishTestPrivate = $false
$global:PublishTestOrigin = ''
$global:PublishTestChanged = $true

# All Git/GitHub commands are mocked; only disposable fixture files are written.
function global:gh {
    $global:LASTEXITCODE = 0
    $global:PublishTestCalls.Add('gh ' + ($args -join ' '))
    if ($args[0] -eq 'api') { return 'fixture-user' }
    if ($args[0] -eq 'repo' -and $args[1] -eq 'view') { return 'https://github.com/wassupjay/indra-ocr-benchmark' }
}
function global:git {
    $global:LASTEXITCODE = 0
    $global:PublishTestCalls.Add('git ' + ($args -join ' '))
    if ($args[0] -eq 'remote' -and $args.Count -eq 1 -and $global:PublishTestOrigin) { return 'origin' }
    if ($args[0] -eq 'remote' -and $args[1] -eq 'get-url') { return $global:PublishTestOrigin }
    if ($args[0] -eq 'diff') {
        if ($args -contains '--quiet') {
            if ($global:PublishTestChanged) { $global:LASTEXITCODE = 1 }
            return
        }
        if ($global:PublishTestPrivate) { return 'eval/benchmarks/private/page_001.txt' }
        return @('README.md', 'publication/MEDIUM_POST.md')
    }
    if ($args[0] -eq 'config' -and $args[1] -eq '--get') { return 'FixtureIdentity' }
}

try {
    New-Item -ItemType Directory -Path (Join-Path $fixture 'publication') -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $fixture '.git') -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $projectRoot 'publish-github.ps1') -Destination $fixture
    $md = Join-Path $fixture 'publication/MEDIUM_POST.md'
    $preview = Join-Path $fixture 'publication/medium-post.html'
    [IO.File]::WriteAllText($md, '[Methods](MEDIUM_FULL_METHODS.md)')
    [IO.File]::WriteAllText($preview, '<a href="MEDIUM_FULL_METHODS.md">Methods</a>')
    & (Join-Path $fixture 'publish-github.ps1') 6>$null | Out-Null
    if (-not ([IO.File]::ReadAllText($md)).Contains('https://github.com/wassupjay/indra-ocr-benchmark/blob/main/publication/MEDIUM_FULL_METHODS.md')) { throw 'Methods URL was not updated.' }
    if (-not ([IO.File]::ReadAllText($preview)).Contains('https://github.com/wassupjay/indra-ocr-benchmark/blob/main/publication/MEDIUM_FULL_METHODS.md')) { throw 'HTML URL was not updated.' }
    if (-not ($global:PublishTestCalls -contains 'git push -u origin main')) { throw 'Expected push was missing.' }
    if ($global:PublishTestCalls -match '--force') { throw 'Unexpected force operation.' }
    if ($global:PublishTestCalls -match 'gh repo create') { throw 'An existing repository must not be recreated.' }

    $global:PublishTestCalls.Clear()
    $global:PublishTestOrigin = 'https://github.com/wassupjay/indra-ocr-benchmark.git'
    $global:PublishTestChanged = $false
    & (Join-Path $fixture 'publish-github.ps1') 6>$null | Out-Null
    if (-not ($global:PublishTestCalls -contains 'git push -u origin main')) { throw 'Retry did not push.' }
    if ($global:PublishTestCalls -match 'git commit|git remote add') { throw 'Retry created a redundant commit or remote.' }

    $global:PublishTestCalls.Clear()
    $global:PublishTestOrigin = 'https://github.com/another/repository.git'
    $rejected = $false
    try { & (Join-Path $fixture 'publish-github.ps1') } catch { $rejected = $_.Exception.Message -match 'another repository' }
    if (-not $rejected -or ($global:PublishTestCalls -match 'git push')) { throw 'Foreign remote was not rejected.' }

    $global:PublishTestCalls.Clear()
    $global:PublishTestOrigin = ''
    $global:PublishTestPrivate = $true
    $rejected = $false
    try { & (Join-Path $fixture 'publish-github.ps1') } catch { $rejected = $_.Exception.Message -match 'Private/generated files' }
    if (-not $rejected -or ($global:PublishTestCalls -match 'gh repo create|git push')) { throw 'Private-file guard failed.' }
    Write-Output 'Publisher tests passed: existing repo, safe retry, foreign-origin rejection and private-data rejection (all mocked).'
} finally {
    Set-Location $previousLocation
    $resolvedFixture = [IO.Path]::GetFullPath($fixture)
    $expectedParent = [IO.Path]::GetFullPath((Join-Path $projectRoot 'eval')) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedFixture.StartsWith($expectedParent, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe fixture cleanup path.' }
    if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture -Recurse -Force }
    Remove-Item Function:\gh, Function:\git -ErrorAction SilentlyContinue
}
