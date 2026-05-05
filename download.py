# https://support.huaweicloud.com/usermanual-dataartsstudio/dataartsstudio_01_1248.html
# 迁移相关的code
from modelarts.session import Session
session = Session()
#session.obs.download_dir(src_obs_dir="obs://eeg-npu/TUEG", dst_local_dir="/home/ma-user/work/BraMT/Raw_data")
#session.obs.download_dir(src_obs_dir="obs://eeg-npu/Raw_data/eeg-motor-movementimagery-dataset-1.0.0", dst_local_dir="/home/ma-user/work/BraMT/Raw_data")
#session.obs.download_dir(src_obs_dir="obs://eeg-npu/Raw_data/physionet.org", dst_local_dir="/home/ma-user/work/BraMT/Raw_data")



#session.obs.download_file(src_obs_file="obs://eeg-npu/Raw_data/BCIC.zip", dst_local_dir="/home/ma-user/work/BraMT/Raw_data/BCIC.zip")
                          

#session.obs.upload_file(src_local_file='/home/ma-user/work/file1.txt', dst_obs_dir='obs://bucket-name/dir1/')

session.obs.upload_dir(src_local_dir="/home/ma-user/work/BraMT-notebook-final",dst_obs_dir="obs://eeg-npu/")