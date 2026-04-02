param(
  [string]$VenvPath = ".venv_app"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $VenvPath)) {
  python -m venv $VenvPath
}

& "$VenvPath\\Scripts\\python.exe" -m pip install --upgrade pip
& "$VenvPath\\Scripts\\pip.exe" install -r requirements_app.txt

# Сборка в папку dist/EmotionResearch/
& "$VenvPath\\Scripts\\pyinstaller.exe" --clean --noconfirm EmotionResearch.spec

Write-Host ""
Write-Host "Готово: dist\\EmotionResearch\\EmotionResearch.exe"

