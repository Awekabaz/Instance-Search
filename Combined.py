import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
import math

_matchTreshold = 5
ratio = 0.75
MAX_IMAGES = 5000
MAX_QUERIES = 20
siftDetector = cv2.xfeatures2d.SIFT_create()
briskDetector = cv2.BRISK_create()
# ===== Crop the image based on bounding box txt file ===== #
def getInstance(img_path):
    try:
        img = cv2.imread(img_path, 0)
        imgBox = img_path.split('.')[0] + '.txt' #get the text file containing bounding box
        f = open(imgBox)
        box = [int(x) for x in next(f).split()] #list contatining box's info
        f.close()
        x, y, w, h = box[0], box[1], box[2], box[3]
        img = img[y:y + h, x:x + w]
    except IOError:
        img = cv2.imread(img_path, 0)
    finally:
        return img

def extractNsaveCombined(idx, image):
    trainImage = getInstance(image)
    kpImg, desImg = siftDetector.detectAndCompute(trainImage, None)
    kpImgB, desImgB = briskDetector.detectAndCompute(trainImage, None)

    print('Features for {} were extracted'.format(idx))
    allDescSift[idx] = desImg
    allDescBrisk[idx] = desImgB

def compareCombinedMethod(qNum, queryImage):
    queryImage = getInstance(queryImage)
    kpQueryB, desQueryB = briskDetector.detectAndCompute(queryImage, None)
    kpQuerySift, desQuerySift = siftDetector.detectAndCompute(queryImage, None)

    for i in range(1, MAX_IMAGES + 1):
        if(allDescSift[i] is None or allDescBrisk is None or desQueryB is None or desQuerySift is None):
            d = 10**8
            print('Query {} | {} number of inloers: {}'.format(queryNumber, image, d))
            resDict[i] = d
            return d

        desResQ = np.append(desQuerySift.flatten(), desQueryB.flatten())
        desResT = np.append(allDescSift[i].flatten(), allDescBrisk[i].flatten())

        ml = max(len(desResQ),len(desResT))

        desResQ = np.concatenate((desResQ , np.zeros(ml-len(desResQ))))
        desResT = np.concatenate((desResT , np.zeros(ml-len(desResT))))

        d = np.linalg.norm(desResQ - desResT)

        print('Query {} | {} score: {}'.format(qNum, i, d))
        resDict[i] = d

if __name__ == '__main__':
    allDescSift= {}
    allDescBrisk = {}

    for i in range(1, MAX_IMAGES + 1):
        trainImg = 'Images/' + str(i).zfill(4) + '.jpg'
        extractNsaveCombined(i, trainImg)

    f = open("rankList.txt", "a")
    for i in range(1,MAX_QUERIES + 1):
        resDict = {}  # does not allow duplicate values
        queryImg = 'Queries/' + str(i).zfill(2) + '.jpg'
        compareCombinedMethod(i, queryImg)
        line = "Q" + str(i) + ": "
        for a, b in sorted(resDict.items(), key=lambda item: item[1], reverse=True):
            line += str(int(b)) + " "
        f.write(line + "\n")
    f.close()

    # compareSift('examples/example_query/01.jpg' , 'Images/0253.jpg')
