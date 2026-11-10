# SayEye CPU 서버를 Cloud Run 에 올린다 (접속 없으면 0대 → 요금 없음).
# 사용: PowerShell 에서 pedapp 폴더로 가서
#   .\deploy\deploy_cpu.ps1 -Project 내프로젝트ID -Key 접속키
# 보호자 연결 · 처음 설정 · 알림 기록(30일)은 Cloud Storage 버킷에 둬서 서버가 다시 켜져도 남는다.
# -Pin 을 안 주면 서버가 확인번호를 한 번 만들어 버킷에 두고 계속 쓴다 (보행자 앱 설정 화면에 보임).
param([Parameter(Mandatory)][string]$Project, [Parameter(Mandatory)][string]$Key, [string]$Pin = "",
      [string]$Region = "asia-northeast3", [string]$Name = "sayeye")
$ErrorActionPreference = "Stop"
gcloud config set project $Project
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com storage.googleapis.com

# 데이터 버킷 (없을 때만 만든다) + 서버 계정이 읽고 쓸 권한
$bucket = "$Project-sayeye-data"
$ErrorActionPreference = "Continue"
gcloud storage buckets describe "gs://$bucket" --format "value(name)" 2>$null | Out-Null
if (-not $?) { gcloud storage buckets create "gs://$bucket" --location $Region --uniform-bucket-level-access --public-access-prevention }
$ErrorActionPreference = "Stop"
$num = gcloud projects describe $Project --format "value(projectNumber)"
gcloud storage buckets add-iam-policy-binding "gs://$bucket" --member "serviceAccount:$num-compute@developer.gserviceaccount.com" --role roles/storage.objectUser | Out-Null

$envs = "SAYEYE_KEY=$Key,DEVICE=cpu,DATA_DIR=/data"
if ($Pin) { $envs += ",GUARDIAN_PIN=$Pin" }
gcloud run deploy $Name --source . --region $Region `
  --cpu 2 --memory 2Gi --min-instances 0 --max-instances 1 --concurrency 20 `
  --timeout 3600 --session-affinity --no-invoker-iam-check --execution-environment gen2 `
  --add-volume "name=data,type=cloud-storage,bucket=$bucket" --add-volume-mount "volume=data,mount-path=/data" `
  --update-env-vars $envs
$url = gcloud run services describe $Name --region $Region --format "value(status.url)"
Write-Host "`n보행자: $url/?k=$Key`n보호자: $url/guardian?k=$Key  (보행자 앱에서 '보호자 연결해줘' → QR 을 보호자 폰으로 찍기)"
