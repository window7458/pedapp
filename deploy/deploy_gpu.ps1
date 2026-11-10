# SayEye GPU 서버 (NVIDIA L4) 를 Cloud Run 에 올린다. 접속이 없으면 0대로 줄어 GPU 요금이 멈춘다.
# 사용: PowerShell 에서 pedapp 폴더로 가서
#   .\deploy\deploy_gpu.ps1 -Project 내프로젝트ID -Key 접속키
# - 보호자 연결 · 처음 설정 · 알림 기록은 CPU 판과 같은 버킷을 써서 그대로 이어진다.
# - 지역은 싱가포르(asia-southeast1): L4 GPU 는 asia-southeast1 · europe-west1/4 · us-central1 · us-east4 만 된다 (도쿄 · 서울 안 됨)
# - 마지막 접속 뒤 몇 분 동안은 켜진 채 기다리다가(그동안은 요금이 나감) 0대로 줄어든다.
param([Parameter(Mandatory)][string]$Project, [Parameter(Mandatory)][string]$Key, [string]$Pin = "",
      [string]$Region = "asia-southeast1", [string]$Name = "sayeye-gpu", [string]$DataRegion = "asia-northeast3")
$ErrorActionPreference = "Stop"
gcloud config set project $Project
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com storage.googleapis.com

# 데이터 버킷 (CPU 판이 이미 만들었으면 그대로 씀)
$bucket = "$Project-sayeye-data"
$ErrorActionPreference = "Continue"
gcloud storage buckets describe "gs://$bucket" --format "value(name)" 2>$null | Out-Null
if (-not $?) { gcloud storage buckets create "gs://$bucket" --location $DataRegion --uniform-bucket-level-access --public-access-prevention }
gcloud artifacts repositories describe sayeye --location $Region 2>$null | Out-Null
if (-not $?) { gcloud artifacts repositories create sayeye --repository-format docker --location $Region }
$ErrorActionPreference = "Stop"
$num = gcloud projects describe $Project --format "value(projectNumber)"
gcloud storage buckets add-iam-policy-binding "gs://$bucket" --member "serviceAccount:$num-compute@developer.gserviceaccount.com" --role roles/storage.objectUser | Out-Null

# CUDA 판 PyTorch 이미지 빌드 (처음엔 15~25분)
$image = "$Region-docker.pkg.dev/$Project/sayeye/sayeye-gpu:latest"
gcloud builds submit --config deploy/cloudbuild-gpu.yaml --substitutions "_IMAGE=$image" .

$envs = "SAYEYE_KEY=$Key,DEVICE=0,DATA_DIR=/data"
if ($Pin) { $envs += ",GUARDIAN_PIN=$Pin" }
gcloud run deploy $Name --image $image --region $Region `
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy --cpu 4 --memory 16Gi --no-cpu-throttling `
  --min-instances 0 --max-instances 1 --concurrency 20 --timeout 3600 --session-affinity --no-invoker-iam-check `
  --add-volume "name=data,type=cloud-storage,bucket=$bucket" --add-volume-mount "volume=data,mount-path=/data" `
  --set-env-vars $envs
$url = gcloud run services describe $Name --region $Region --format "value(status.url)"
Write-Host "`nGPU 보행자: $url/?k=$Key`nGPU 보호자: $url/guardian?k=$Key  (보행자 앱에서 '보호자 연결해줘' → QR 을 보호자 폰으로 찍기)"
