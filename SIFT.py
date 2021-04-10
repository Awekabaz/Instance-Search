import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
import math

_matchTreshold = 5
ratio = 0.75
resDict = {}
MAX_IMAGES = 5000
MAX_QUERIES = 20
siftDetector = cv2.xfeatures2d.SIFT_create()

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

def extractNsave(idx, image):
    trainImage = getInstance(image)
    kpImg, desImg = siftDetector.detectAndCompute(trainImage, None)
    print('Features for {} were extracted'.format(idx))
    allDesc[idx] = desImg
    allKps[idx] = kpImg

def compareSift(qNum, queryImage):
    queryImage = getInstance(queryImage)

    # detect the keypoints and descriptors with SIFT
    kpQuery, desQuery = siftDetector.detectAndCompute(queryImage, None)

    # 2 dictionaries for FLANN based matcher
    indexParams = dict(algorithm = 0, trees = 5)
    searchParams = dict(checks=100)
    for i in range(1, MAX_IMAGES + 1):
        fMatcher = cv2.FlannBasedMatcher(indexParams, searchParams)
        matches = fMatcher.knnMatch(desQuery,allDesc[i],k=2)

        # store good matches according to Lowe's ratio test
        rtMatches = []
        for a, b in matches:
            if a.distance < ratio*b.distance:
                rtMatches.append(a)

        if len(rtMatches)>_matchTreshold:
            ptsQ = np.float32([kpQuery[x.queryIdx].pt for x in rtMatches])
            ptsI = np.float32([allKps[i][x.trainIdx].pt for x in rtMatches])

            # Reshape for findHomography() function
            ptsQ = ptsQ.reshape(-1,1,2)
            ptsI = ptsI.reshape(-1,1,2)

            M, mask = cv2.findHomography(ptsQ, ptsI, cv2.RANSAC,5.0)
            inliers = mask.ravel().tolist()
        else:
            inliers = None

        res = 0
        if inliers:
            for x in inliers:
                if x == 1:
                    res += 1
        print('Query {} | {} number of inliers: {}'.format(qNum, i, res))
        resDict[i] = res

if __name__ == '__main__':
    allDesc = {}
    allKps = {}
    # Extracting all the keypoints and descriptors for all 5000 images
    # Save to dictionaries for further faster comparison
    for i in range(1, MAX_IMAGES + 1):
        trainImg = 'Images/' + str(i).zfill(4) + '.jpg'
        extractNsave(i, trainImg)

    f = open("rankList.txt", "a")
    for i in range(1,MAX_QUERIES + 1):
        resDict = {}  # does not allow duplicate values
        queryImg = 'Queries/' + str(i).zfill(2) + '.jpg'
        compareSift(i, queryImg)
        line = "Q" + str(i) + ": "
        for a, b in sorted(resDict.items(), key=lambda item: item[1], reverse=True):
            line += str(int(b)) + " "
        f.write(line + "\n")
    f.close()
