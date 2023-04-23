import cv2
import numpy as np
image = cv2.imread("../data/SRVD_data/raw_noisy/MOT17-09_raw/000001_raw_iso12800_noisy0.tiff", -1)
crvd_gt = cv2.imread("../data/CRVD/indoor_raw_gt/scene1/ISO12800/frame1_clean_and_slightly_denoised.tiff", -1).astype(np.uint16)
# save_npy = np.save("test_path.npy", image)
# save_npy = np.save("CRVD_gt.npy", image)
crvd_gt_npy = np.save("crvd_gt.npy", crvd_gt)
# NO Space saving
import time
st = time.time()

for i in np.arange(100):
    crvd_gt_npy_load = np.load("test_path.npy")
    # print(crvd_gt_npy_load.shape)
print('np.load time', time.time()-st)

for i in np.arange(100):
    image = cv2.imread("../data/SRVD_data/raw_noisy/MOT17-09_raw/000001_raw_iso12800_noisy0.tiff", -1)
    # print(crvd_gt.shape)
print('cv2.imread time', time.time()-st)
