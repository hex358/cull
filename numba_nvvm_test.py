from numba import cuda
from numba.cuda.cuda_paths import get_cuda_paths
import numpy as np
import os

print("CUDA_PATH =", os.environ.get("CUDA_PATH"))
print("CUDA_HOME =", os.environ.get("CUDA_HOME"))
print("NUMBAPRO_NVVM =", os.environ.get("NUMBAPRO_NVVM"))
print("NUMBAPRO_LIBDEVICE =", os.environ.get("NUMBAPRO_LIBDEVICE"))
print("cuda paths =", get_cuda_paths())

@cuda.jit
def k(a):
	i = cuda.grid(1)
	if i < a.size:
		a[i] += 1

a = cuda.to_device(np.zeros(16, dtype=np.int32))
k[1, 32](a)
cuda.synchronize()
print("numba compile OK:", a.copy_to_host()[:4])