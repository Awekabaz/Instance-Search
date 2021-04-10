import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
import math

MAX_IMAGES = 5000
MAX_QUERIES = 20

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

def chi2_distance(hist1, hist2, eps = 1e-10):
    temp = [((x1-x2)**2)/(x1+x2+eps) for (x1,x2) in zip(hist1, hist2)]
    dist = 0.5*np.sum(temp)
    return dist

def numpyArraysEuclidianDist(a1, a2):
    dist = np.sum([(x2-x1)**2 for (x1, x2) in zip(a1, a2)]) **0.5
    return dist

def distanceMetric(idx, queryImg):
    queryImg = getInstance(queryImg)
    queryHist = buildHist(queryImg)

    for i in range(1, MAX_IMAGES + 1):
        distance = chi2_distance(queryHist, allHist[i])
        print('Query {} | {} number of inliers: {}'.format(idx, i, distance))
        resDict[i] = distance


def genHists(idx, img):
    img = getInstance(img)
    hist = buildHist(img)
    allHist[idx] = hist

def buildHist(img):
    rows, columns, channels = img.shape

    rtnHist = np.zeros([16 * 16 * 16, 1], np.float32)
    binSize = 16
    for r in range(rows):
        for col in range(columns):
            blue = img[r, col, 0]
            green = img[r, col, 1]
            red = img[r, col, 2]
            index = int(blue/binSize)*256+int(green/binSize)*16+int(red/binSize)
            rtnHist[int(index), 0] += 1

    return rtnHist

if __name__ == '__main__':
    allHist = {}
    resDict = {}
    for i in range(1, MAX_IMAGES + 1):
        trainImg = 'Images/' + str(i).zfill(4) + '.jpg'
        genHists(i, trainImg)

    f = open("rankList.txt", "a")
    for i in range(1,MAX_QUERIES + 1):
        resDict = {}  # does not allow duplicate values
        queryImg = 'Queries/' + str(i).zfill(2) + '.jpg'
        distanceMetric(i, queryImg)
        line = "Q" + str(i) + ": "
        for a, b in sorted(resDict.items(), key=lambda item: item[1], reverse=False):
            line += str(int(b)) + " "
        f.write(line + "\n")
    f.close()

    # compareSift('examples/example_query/01.jpg' , 'Images/0253.jpg')
