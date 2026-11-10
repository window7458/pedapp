# SayEye 풍경 서버 ("주변 설명해줘", SJ 말로 듣는 풍경 · Qwen3.5-9B) 를 Cloud Run GPU(L4) 에 올리고 앱 서버와 잇는다.
# 사용: PowerShell 에서 pedapp 폴더로 가서
#   .\deploy\deploy_scene.ps1 -Project 내프로젝트ID              (처음)
#   .\deploy\deploy_scene.ps1 -Project 내프로젝트ID -SkipBuild   (이미지는 그대로, 설정만 다시)
# - 앱 서버(sayeye-gpu)만 부를 수 있게 닫아 둔다 (밖에서 아무나 GPU 를 쓰지 못하게). 앱 서버 계정에 호출 권한을 준다.
# - 모델(17.6 GB)은 저장 버킷에 한 번만 넣고 거기서 읽는다 (Cloud Run 디스크는 메모리라 받아 두면 메모리 한도를 넘음).
# - 접속이 없으면 0대 → 요금 없음. 보행자가 "시작"하면 앱 서버가 미리 깨운다 (켜지는 데 몇 분).
param([Parameter(Mandatory)][string]$Project, [string]$Region = "asia-southeast1",
      [string]$Name = "sayeye-scene", [string]$AppService = "sayeye-gpu", [switch]$SkipBuild)
$ErrorActionPreference = "Stop"
gcloud config set project $Project
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
$bucket = "$Project-sayeye-data"

$ErrorActionPreference = "Continue"
gcloud artifacts repositories describe sayeye --location $Region 2>$null | Out-Null
if (-not $?) { gcloud artifacts repositories create sayeye --repository-format docker --location $Region }
# 0) 모델 파일을 버킷에 (없을 때만, 15~30분)
gcloud storage ls "gs://$bucket/models/Qwen3.5-9B/config.json" 2>$null | Out-Null
$haveModel = $?
$ErrorActionPreference = "Stop"
if (-not $haveModel) {
  Write-Host "모델을 저장 버킷에 넣는 중 (처음 한 번)..."
  gcloud builds submit --no-source --config scene_server/model_to_bucket.yaml --substitutions "_BUCKET=$bucket"
}

# 1) 이미지 빌드 (scene_server 폴더만 올린다)
$image = "$Region-docker.pkg.dev/$Project/sayeye/sayeye-scene:latest"
if (-not $SkipBuild) { gcloud builds submit scene_server --tag $image --machine-type e2-highcpu-8 --timeout 3600s }

# 2) 배포 — 공개하지 않음(IAM 호출 검사 켜 둠). 메모리 16 GB (L4 최소), 모델은 버킷을 /data 에 붙여 읽음
gcloud run deploy $Name --image $image --region $Region `
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy --cpu 4 --memory 16Gi --no-cpu-throttling `
  --min-instances 0 --max-instances 1 --concurrency 8 --timeout 600 --port 8080 --invoker-iam-check `
  --add-volume "name=data,type=cloud-storage,bucket=$bucket,readonly=true" --add-volume-mount "volume=data,mount-path=/data" `
  --update-env-vars "MODEL_PATH=/data/models/Qwen3.5-9B"

# 3) 앱 서버 계정만 부를 수 있게
$num = gcloud projects describe $Project --format "value(projectNumber)"
gcloud run services add-iam-policy-binding $Name --region $Region `
  --member "serviceAccount:$num-compute@developer.gserviceaccount.com" --role roles/run.invoker | Out-Null

# 4) 앱 서버에 풍경 서버 주소 알려 주기 (앱 서버는 다시 빌드하지 않고 설정만 바뀜)
$url = gcloud run services describe $Name --region $Region --format "value(status.url)"
gcloud run services update $AppService --region $Region --update-env-vars "SCENE_URL=$url"
Write-Host "`n풍경 서버: $url  (앱 서버 $AppService 에 SCENE_URL 로 연결됨)"
