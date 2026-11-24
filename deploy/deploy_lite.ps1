# SayEye 가벼운 서버판(lite) 을 따로 올린다 — 기존 sayeye-gpu 는 그대로 두고 sayeye-lite 로 시험.
# 사용: PowerShell 에서 pedapp 폴더로 가서   .\deploy\deploy_lite.ps1
# 값(프로젝트 · 접속 키 · 공공데이터 · TMAP · KRIC 키)은 이 PC 의 C:\Users\windo\.sayeye_deploy.txt 에서 읽는다 (저장소에 안 올라감).
# 먼저 한 번: python code\app\tools\export_onnx.py  (weights 의 .pt → .onnx)
param([int]$MinInstances = 0, [string]$Name = "sayeye-lite", [string]$Region = "asia-southeast1")
$ErrorActionPreference = "Stop"
$cfg = @{}
Get-Content "$env:USERPROFILE\.sayeye_deploy.txt" -Encoding UTF8 | ForEach-Object {
  if ($_ -match '^\s*([A-Za-z_]+)\s*=\s*(.+?)\s*$') { $cfg[$matches[1]] = $matches[2] }
}
$Project = $cfg["Project"]; $Key = $cfg["Key"]
if (-not $Project -or -not $Key) { throw "C:\Users\windo\.sayeye_deploy.txt 에 Project= 와 Key= 가 있어야 해요" }
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($w in "final_10class_best.onnx", "yolo11n.onnx", "yolo11n_tactile.onnx") {
  if (-not (Test-Path "code\app\weights\$w")) { throw "code\app\weights\$w 가 없어요. python code\app\tools\export_onnx.py 를 먼저 실행해 주세요" }
}
gcloud config set project $Project

$image = "$Region-docker.pkg.dev/$Project/sayeye/sayeye-lite:latest"
gcloud builds submit --config deploy/cloudbuild-lite.yaml --substitutions "_IMAGE=$image" .

$bucket = "$Project-sayeye-data"
$envs = "SAYEYE_KEY=$Key,DEVICE=0,DATA_DIR=/data,INFER=onnx,STT=0"
foreach ($k in "DATA_GO_KR_KEY", "TMAP_KEY", "KRIC_KEY") { if ($cfg[$k]) { $envs += ",$k=$($cfg[$k])" } }
$ErrorActionPreference = "Continue"
$svc = gcloud run services describe sayeye-gpu --region $Region --format json 2>$null | ConvertFrom-Json   # 풍경 서버 주소는 기존 판 설정에서
$scene = if ($svc) { ($svc.spec.template.spec.containers[0].env | Where-Object { $_.name -eq "SCENE_URL" }).value } else { $null }
$ErrorActionPreference = "Stop"
if ($scene) { $envs += ",SCENE_URL=$scene" }
gcloud run deploy $Name --image $image --region $Region `
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy --cpu 4 --memory 16Gi --no-cpu-throttling `
  --min-instances $MinInstances --max-instances 1 --concurrency 20 --timeout 3600 --session-affinity --no-invoker-iam-check `
  --add-volume "name=data,type=cloud-storage,bucket=$bucket" --add-volume-mount "volume=data,mount-path=/data" `
  --update-env-vars $envs
$url = gcloud run services describe $Name --region $Region --format "value(status.url)"
Write-Host "`n가벼운 서버판: $url/?k=$Key`n(기존 판은 그대로: sayeye-gpu)"
