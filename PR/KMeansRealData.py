import cv2
from sklearn.cluster import KMeans
import numpy as np
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture image")
    exit()
    
frame = cv2.resize(frame, (200,200))

pixels = frame.reshape((-1,3))

k = 3
knn = KMeans(n_clusters = k)
knn.fit(pixels)

clustered = knn.cluster_centers_[knn.labels_]
clustered_img = clustered.reshape(frame.shape).astype(np.uint8)

cv2.imshow("original Image",frame)
cv2.imshow('K-Means Clustered Image', clustered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()