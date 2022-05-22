import subprocess
import time
datasets = ["GraIL-BM_fb237_v1", "GraIL-BM_fb237_v2", "GraIL-BM_fb237_v3", "GraIL-BM_fb237_v4", 
    "GraIL-BM_nell_v1", "GraIL-BM_nell_v2", "GraIL-BM_nell_v3", "GraIL-BM_nell_v4",
    "GraIL-BM_WN18RR_v1", "GraIL-BM_WN18RR_v2", "GraIL-BM_WN18RR_v3", "GraIL-BM_WN18RR_v4"]
'''
for ds in datasets:
    subprocess.call(f"python negative_sampling.py --dataset {ds} --sample_rate 3 &", shell=True)
'''

for i, ds in enumerate(datasets[:3]):
    subprocess.call(f"CUDA_VISIBLE_DEVICES={1} nohup python train.py --dataset {ds} >> out/{ds}_{i} 2>&1 &", shell=True)
