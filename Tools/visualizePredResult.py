
predictDir = "/home/hxie1/data/OCT_Beijing/numpy/predict"

numPatients = 8
numSlices = 31

import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    for patientID in range(0, numPatients):
        for sliceID in range(0, numSlices):
            fileSuffix =  f"_p{patientID:02d}_s{sliceID:02d}.npy"
            image = np.load(os.path.join(predictDir,f"TestImage"+fileSuffix))
            gt = np.load(os.path.join(predictDir,f"TestGT"+fileSuffix))
            gt_n1 = np.load(os.path.join(predictDir, f"TestGT_n1" + fileSuffix))
            gt_p1 = np.load(os.path.join(predictDir, f"TestGT_p1" + fileSuffix))
            pred = np.load(os.path.join(predictDir, f"TestPred" + fileSuffix))
            (H,W) = image.shape

            f = plt.figure()
            plt.imshow(image, cmap='gray')
            plt.plot(range(0, W), gt, 'g,')
            plt.plot(range(0, W), gt_n1, 'b,')
            plt.plot(range(0, W), gt_p1, 'y,')
            plt.plot(range(0, W), pred, 'r,')

            titleName = f"patient:{patientID:02d}, slice:{sliceID:02d}, Blue: GT-1, Green:GT, Yellow: GT+1, Red:Prediction"
            plt.title(titleName)

            plt.savefig(os.path.join(predictDir, f"TestResult_{patientID:02d}_{sliceID:02d}" + ".png"))
            plt.close()

if __name__ == "__main__":
    main()
