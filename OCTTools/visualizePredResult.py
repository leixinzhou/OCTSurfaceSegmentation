
predictDir = "/home/hxie1/data/OCT_Beijing/numpy/predict"

numPatients = 8
numSlices = 31

import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    for patientID in range(0, numPatients):
        for sliceID in range(0, numSlices):
            fileSuffix =  f"{patientID:02d}_{sliceID:02d}.npy"
            image = np.load(os.path.join(predictDir,f"TestImage"+fileSuffix))
            gt = np.load(os.path.join(predictDir,f"TestGT"+fileSuffix))
            pred = np.load(os.path.join(predictDir, f"TestPred" + fileSuffix))
            (H,W) = image.shape

            f = plt.figure()
            plt.imshow(image, cmap='gray')
            plt.plot(range(0, W), gt, 'g,',linewidth=10)
            plt.plot(range(0, W), pred, 'r,', linewidth=10)
            titleName = f"patient:{patientID:02d}, slice:{sliceID:02d}, Green:GT, Red:Prediction"
            plt.title(titleName)

            plt.savefig(os.path.join(predictDir, f"TestResult_{patientID:02d}_{sliceID:02d}" + ".png"))
            plt.close()

if __name__ == "__main__":
    main()
