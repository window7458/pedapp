# SayEye 전부 한 번에 배포 — 경고 서버(sayeye-gpu) + 풍경 서버(sayeye-scene).
# 사용: PowerShell 에서 pedapp 폴더로 가서   .\deploy\deploy_all.ps1
# 값(프로젝트 · 접속 키 · 공공데이터 키 · TMAP 키 · KRIC 키)은 이 PC 의 C:\Users\windo\.sayeye_deploy.txt 에서 읽는다 (저장소에 안 올라감).
# 순서: ① 고장 난 풍경 서버가 메모리를 잡고 있으면 지움 → ② 경고 서버 → ③ 풍경 서버(모델은 처음 한 번 버킷에)
# 메모리 한도(싱가포르 40 GB) 때문에 두 서버를 동시에 바꾸지 않고 차례로 올린다.
param([int]$MinInstances = 0, [switch]$Scene)
$ErrorActionPreference = "Stop"
$cfg = @{}
Get-Content "$env:USERPROFILE\.sayeye_deploy.txt" -Encoding UTF8 | ForEach-Object {
  if ($_ -match '^\s*([A-Za-z_]+)\s*=\s*(.+?)\s*$') { $cfg[$matches[1]] = $matches[2] }
}
$Project = $cfg["Project"]; $Key = $cfg["Key"]
if (-not $Project -or -not $Key) { throw "C:\Users\windo\.sayeye_deploy.txt 에 Project= 와 Key= 가 있어야 해요" }
Set-Location (Split-Path $PSScriptRoot -Parent)

# ① 준비되지 않은(실패한) 풍경 서버가 남아 있으면 지운다 — 메모리를 잡고 있어 다른 배포를 막는다
$ErrorActionPreference = "Continue"
$ready = gcloud run services describe sayeye-scene --region asia-southeast1 --project $Project --format "value(status.conditions[0].status)" 2>$null
if ($? -and $ready -ne "True") { gcloud run services delete sayeye-scene --region asia-southeast1 --project $Project --quiet }
$ErrorActionPreference = "Stop"

# ② 경고 서버 (앱)
$extra = @{}
if ($cfg["DATA_GO_KR_KEY"]) { $extra["DataKey"] = $cfg["DATA_GO_KR_KEY"] }
& "$PSScriptRoot\deploy_gpu.ps1" -Project $Project -Key $Key -MinInstances $MinInstances @extra
$envs = @()
if ($cfg["TMAP_KEY"]) { $envs += "TMAP_KEY=$($cfg['TMAP_KEY'])" }
if ($cfg["KRIC_KEY"]) { $envs += "KRIC_KEY=$($cfg['KRIC_KEY'])" }
if ($envs.Count) { gcloud run services update sayeye-gpu --region asia-southeast1 --project $Project --update-env-vars ($envs -join ",") }

# ③ 풍경 서버 ("주변 설명해줘") — 바뀐 게 없고 잘 돌고 있으면 건너뜀 (매번 5분씩 다시 만들던 것). 꼭 다시 하려면 -Scene
$sceneFiles = Get-ChildItem scene_server -Recurse -File | Sort-Object FullName
$sceneHash = (($sceneFiles | ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash }) + (Get-FileHash "$PSScriptRoot\deploy_scene.ps1").Hash) -join ""
$sceneHash = [System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes($sceneHash))) -replace "-", ""
$hashFile = "$env:USERPROFILE\.sayeye_scene_hash"
$lastHash = if (Test-Path $hashFile) { (Get-Content $hashFile -Raw).Trim() } else { "" }
$ErrorActionPreference = "Continue"
$sceneReady = gcloud run services describe sayeye-scene --region asia-southeast1 --project $Project --format "value(status.conditions[0].status)" 2>$null
$ErrorActionPreference = "Stop"
if ($Scene -or $sceneHash -ne $lastHash -or $sceneReady -ne "True") {
  & "$PSScriptRoot\deploy_scene.ps1" -Project $Project
  Set-Content $hashFile $sceneHash -Encoding ASCII
} else {
  Write-Host "풍경 서버는 바뀐 게 없어 건너뜀 (다시 올리려면 -Scene)"
}

$url = gcloud run services describe sayeye-gpu --region asia-southeast1 --project $Project --format "value(status.url)"
Write-Host "`n다 올렸어요. 보행자 앱: $url/?k=$Key"
