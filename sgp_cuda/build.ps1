$ErrorActionPreference = "Stop"
cd D:\aeroo\pythonProject\.venv\dir\sgp_cuda
Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item _skbuild -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item *.egg-info -Recurse -Force -ErrorAction SilentlyContinue
python -m pip install -e . --no-build-isolation -v