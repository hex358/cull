param(
  [string]$CudaRoot = $env:CUDA_PATH
)

if (-not $CudaRoot) {
  $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
}

if (-not (Test-Path $CudaRoot)) {
  throw "CUDA root not found: $CudaRoot"
}

$nvvmDll = Get-ChildItem "$CudaRoot\nvvm\bin" -Recurse -Include "nvvm.dll","nvvm64*.dll" -ErrorAction Stop | Select-Object -First 1
$libDevice = Get-ChildItem "$CudaRoot\nvvm\libdevice" -Recurse -Filter "libdevice*.bc" -ErrorAction Stop | Select-Object -First 1

if (-not $nvvmDll) {
  throw "NVVM DLL not found under $CudaRoot\nvvm\bin"
}

if (-not $libDevice) {
  throw "libdevice not found under $CudaRoot\nvvm\libdevice"
}

$env:CUDA_PATH = $CudaRoot
$env:CUDA_HOME = $CudaRoot
$env:NUMBAPRO_NVVM = $nvvmDll.FullName
$env:NUMBAPRO_LIBDEVICE = $libDevice.DirectoryName
$env:PATH = "$CudaRoot\bin;$CudaRoot\nvvm\bin;$env:PATH"

Write-Host "[ORBIT CUDA env]"
Write-Host "  CUDA_PATH=$env:CUDA_PATH"
Write-Host "  CUDA_HOME=$env:CUDA_HOME"
Write-Host "  NUMBAPRO_NVVM=$env:NUMBAPRO_NVVM"
Write-Host "  NUMBAPRO_LIBDEVICE=$env:NUMBAPRO_LIBDEVICE"
