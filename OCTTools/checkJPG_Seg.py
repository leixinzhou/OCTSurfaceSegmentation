# check JPG and Segmentation correspondence.

import os
import xml.etree.ElementTree as ET
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from FileUtilities import *

volumesDir = "/home/hxie1/data/OCT_Beijing/control"
segsDir = "/home/hxie1/data/OCT_Beijing/Correcting_Seg"

def main():
    volumesList = glob.glob(volumesDir+f"/*_Volume")
    for volumeDir in volumesList:
        patient = os.path.basename(volumeDir)
        segFile = os.path.join(segsDir, patient + "_Sequence_Surfaces_Iowa.xml")

        surfacesArray = getSurfacesArray(segFile)
        Z,surface_num, X = surfacesArray.shape

        imagesList = glob.glob(volumeDir+f"/*_OCT[0-3][0-9].jpg")
        imagesList.sort()

        if Z != len(imagesList):
           print(f"Error: at {volumesList}, the slice number does not match png files.")
           return

        for z in range(0, Z):
            imageFile = imagesList[z]
            pathStem = os.path.splitext(imageFile)[0]
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            image = mpimg.imread(imageFile)
            f = plt.figure()
            plt.imshow(image, cmap='gray')
            for s in range(0, surface_num):
                plt.plot(range(0,X), surfacesArray[z,s,:], linewidth=0.7)
            titleName = patient + f"_OCT{z+1:02d}_surface"
            plt.title(titleName)

            plt.savefig(os.path.join(volumeDir, titleName+".png"))
            plt.close()
        print(f"output volume surface for {volumeDir}")

    print("============End of Program=============")


if __name__ == "__main__":
    main()