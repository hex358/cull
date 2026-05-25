$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:Path="$env:CUDA_PATH\bin;$env:Path"

cmd /c "`"D:\vs\VC\Auxiliary\Build\vcvars64.bat`" && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        Set-Item -Path "Env:\$($matches[1])" -Value $matches[2]
    }
}

Write-Host "CUDA:"
nvcc --version

Write-Host "MSVC:"
where.exe cl