# Instance-Search
Implementation of different methods for instance search

I implemented an application that can perform instance search using SIFT and Color Histogram methods. I also tried to combine the SIFT and BRISK descriptors to solve the problem.
_______________________________________________________________________

The instance of an image object is given as bounding box. The bounding box is represented as (top left corner in x, top left corner in y, width, height) in the text file in the same directory as images, both for query and train images.

**`getInstance()`** function implements the image cropping if there is a corresponding bounding box file. It takes the image path shown above and returns the image for further processing.

_______________________________________________________________________

## SIFT Algorithm
1. Iterate over all train images extract the keypoints and descriptor vectors. Pass image path to **`extractNsave(idx, img)`** which saves the Key Points/Descriptors to 2 dictionaries for easier retrievial.

2. Iterate over all query images, extract the keypoints and descriptor vectors. For each query match the desriptors.

3. Matching. **FLANN based mathcer** was used to match the descriptor vectors and further **knnMatch** was performed (Euclidian distance as metric). Also, mathes were filtered with Lowe's ratio test (resulted in 0.72). Main helper function **`compareSift(qNum, aueryImg)`** does all the calculations.

4. Outlier rejection
There can be some possible errors while matching which may affect the result. Pass the set of points from both the sets to **`cv2.findHomography(ptsQ, ptsI, cv2.RANSAC,5.0)`**. The algorithm uses RANSAC with 99.5% confidence applied to reject the outliers and to find the Homography Matrix (which were not used) and return a mask which specifies the inlier and outlier points. More inliers the sets have, the more similar they are

**SIFT RESULTS `(0.88 and 0.93 accuracy)`**
![query2](/results/q2-1.png)
![query3](/results/q3-1.png)
