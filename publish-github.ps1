$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$repository = 'wassupjay/indra-ocr-benchmark'
$expectedRepositoryUrl = "https://github.com/$repository"

$githubLogin = & gh api user --jq .login
if ($LASTEXITCODE -ne 0 -or -not $githubLogin) {
    throw 'GitHub access failed in this terminal. Check its network connection and gh auth status before retrying.'
}
$repositoryUrl = & gh repo view $repository --json url --jq .url
if ($LASTEXITCODE -ne 0 -or $repositoryUrl -ne $expectedRepositoryUrl) {
    throw 'Could not verify access to wassupjay/indra-ocr-benchmark. No push was attempted.'
}

if (-not (Test-Path -LiteralPath '.git')) {
    & git init -b main
    if ($LASTEXITCODE -ne 0) { throw 'Could not initialize Git.' }
}
$existingRemotes = @(& git remote)
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect Git remotes.' }
$originPresent = $existingRemotes -contains 'origin'
if ($originPresent) {
    $originUrl = & git remote get-url origin
    if ($LASTEXITCODE -ne 0 -or $originUrl -notin @($repositoryUrl, "$repositoryUrl.git", "git@github.com:$repository.git")) {
        throw 'Origin points to another repository. No remote was changed.'
    }
}

# Review the actual Git selection before publishing any files.
& git add --all
if ($LASTEXITCODE -ne 0) { throw 'Could not stage the repository files.' }
$stagedFiles = @(& git diff --cached --name-only)
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect staged files.' }
$privateFiles = @($stagedFiles | Where-Object {
    $_ -match '(^|/)(\.env($|\.)|credentials\.json$|secrets\.json$)' -or
    $_ -match '\.(pdf|pth|onnx|whl|pem|key)$' -or
    $_ -match '(^|/)(extracted\.md|Rich_Dad_Poor_Dad_transcription\.md|WORK_SUMMARY\.md)$' -or
    $_ -match '^(\.benchmark-venv|\.easyocr-venv|\.ocr-models|\.onnx-models)/' -or
    $_ -match '^eval/(?!config\.json$)'
})
if ($privateFiles.Count) {
    throw ('Private/generated files are staged; unstage them first: ' + ($privateFiles -join ', '))
}

foreach ($setting in @(@('user.name', $githubLogin), @('user.email', "$githubLogin@users.noreply.github.com"))) {
    $configuredValue = & git config --get $setting[0]
    if (-not $configuredValue) {
        & git config --local $setting[0] $setting[1]
        if ($LASTEXITCODE -ne 0) { throw "Could not configure $($setting[0])." }
    }
}

$methodsUrl = "$repositoryUrl/blob/main/publication/MEDIUM_FULL_METHODS.md"
$utf8 = New-Object System.Text.UTF8Encoding($false)
$markdown = Join-Path $PSScriptRoot 'publication/MEDIUM_POST.md'
$preview = Join-Path $PSScriptRoot 'publication/medium-post.html'
[IO.File]::WriteAllText($markdown, ([IO.File]::ReadAllText($markdown)).Replace('(MEDIUM_FULL_METHODS.md)', "($methodsUrl)"), $utf8)
[IO.File]::WriteAllText($preview, ([IO.File]::ReadAllText($preview)).Replace('href="MEDIUM_FULL_METHODS.md"', "href=`"$methodsUrl`""), $utf8)

& git add --all
if ($LASTEXITCODE -ne 0) { throw 'Could not stage the final article links.' }
& git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    & git commit -m 'Publish audited CPU OCR benchmark and visual results'
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed. No push was attempted.' }
} elseif ($LASTEXITCODE -ne 0) {
    throw 'Could not inspect the staged changes. No push was attempted.'
}
& git branch -m main
if ($LASTEXITCODE -ne 0) { throw 'Could not select the main branch.' }
if (-not $originPresent) {
    & git remote add origin "$repositoryUrl.git"
    if ($LASTEXITCODE -ne 0) { throw 'Could not add the remote.' }
}
& gh auth setup-git
if ($LASTEXITCODE -ne 0) { throw 'GitHub Git authentication setup failed.' }
& git push -u origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed. Your local commit is intact; do not use a force push.' }

Write-Host "Published: $repositoryUrl"
Write-Host "Medium methods link: $methodsUrl"
