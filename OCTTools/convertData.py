
import glob as glob
import os
import sys
from FileUtilities import *
import random
import numpy as np
from imageio import imread


k = 0
K = 6  # K-fold Cross validation, the k-fold is for test, (k+1)%K is validation, others for training.

W = 768
H = 496
NumSurfaces = 11
NumSlices = 31  # for each patient

volumesDir = "/home/hxie1/data/OCT_Beijing/control"
segsDir = "/home/hxie1/data/OCT_Beijing/Correcting_Seg"

outputDir = "/home/hxie1/data/OCT_Beijing/numpy"
patientsListFile = os.path.join(outputDir, "patientsList.txt")

def saveVolumeSurfaceToNumpy(volumesList, goalImageFile, goalSurfaceFile):
    allPatientsImageArray = np.empty((len(volumesList)*NumSlices,H, W), dtype=np.float)
    allPatientsSurfaceArray = np.empty((len(volumesList)*NumSlices, NumSurfaces, W),dtype=np.int)

    s = 0 # initial slice for each patient
    for volume in volumesList:
        segFile = os.path.basename(volume)+"_Sequence_Surfaces_Iowa.xml"
        segFile = os.path.join(segsDir, segFile)

        surfacesArray = getSurfacesArray(segFile)
        Z,surfaces_num, X = surfacesArray.shape
        assert X == W and surfaces_num == NumSurfaces
        allPatientsSurfaceArray[s:s+Z,:,:] = surfacesArray

        # read image data
        imagesList = glob.glob(volume + f"/*_OCT[0-3][0-9].jpg")
        imagesList.sort()
        if Z != len(imagesList):
           print(f"Error: at {volumesList}, the slice number does not match png files.")
           return

        for z in range(0, Z):
            allPatientsImageArray[s,] = imread(imagesList[z])
            s +=1
    # save
    np.save(goalImageFile, allPatientsImageArray)
    np.save(goalSurfaceFile, allPatientsSurfaceArray)

def main():
    # get files list
    if os.path.isfile(patientsListFile):
        patientsList = loadInputFilesList(patientsListFile)
    else:
        patientsList = glob.glob(volumesDir + f"/*_Volume")
        patientsList.sort()
        random.seed(201910)
        random.shuffle(patientsList)
        saveInputFilesList(patientsList, patientsListFile)

    # split files in sublist
    N = len(patientsList)
    patientsSubList= []
    step = N//K
    for i in range(0,N, step):
        nexti = i + step
        if nexti + step > N:
            nexti = N
        patientsSubList.append(patientsList[i:nexti])
        if nexti == N:
            break

    # partition
    partitions = {}
    partitions["test"] = patientsSubList[k]
    k1 = (k + 1) % K  # validation k
    partitions["validation"] = patientsSubList[k1]
    partitions["training"] = []
    for i in range(K):
        if i != k and i != k1:
            partitions["training"] += patientsSubList[i]

    # save to file
    saveVolumeSurfaceToNumpy(partitions["test"], os.path.join(outputDir, 'test', f"images_CV{k}.npy"),\
                                                 os.path.join(outputDir, 'test', f"surfaces_CV{k}.npy"))
    saveVolumeSurfaceToNumpy(partitions["validation"], os.path.join(outputDir, 'validation', f"images_CV{k}.npy"), \
                                                       os.path.join(outputDir, 'validation', f"surfaces_CV{k}.npy"))
    saveVolumeSurfaceToNumpy(partitions["training"], os.path.join(outputDir, 'training', f"images_CV{k}.npy"), \
                                                     os.path.join(outputDir, 'training', f"surfaces_CV{k}.npy"))


    print(f"test: {len(partitions['test'])} patients;  validation: {len(partitions['validation'])} patients;  training: {len(partitions['training'])} patients, ")
    print("===End of prorgram=========")

if __name__ == "__main__":
    main()