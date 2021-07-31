import cv2
import glob
from operator import itemgetter
img_query = cv2.imread('query.jpg', 0)
sift = cv2.xfeatures2d.SIFT_create()
keypoints_first, distance_first = sift.detectAndCompute(img_query, None)

bf_matcher = cv2.BFMatcher()

matches = []
images = glob.glob("tiny_data/*.jpg")
for img in images:
    sum = 0
    image = cv2.imread(img)
    keypoints_second, distance_second = sift.detectAndCompute(image, None)
    match = bf_matcher.match(queryDescriptors=distance_first, trainDescriptors=distance_second)
    match = sorted(match, key=lambda x: x.distance)
    for i in range(10):
        sum += match[i].distance
    matches.append((sum, image))
matches = sorted(matches, key=lambda x:x[0])
file = open("Distances.txt", "a")
for match in matches:
    file.write(str(match[0])+'\n')
cv2.imshow("Best", matches[0][1])
cv2.waitKey(0)
