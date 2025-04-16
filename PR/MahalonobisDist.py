from scipy.spatial import ConvexHull
from scipy.spatial.distance import mahalanobis
import numpy as np

def calculate_centroid(polygon):
    return(np.mean(polygon, axis = 0))

def calculate_covariance_matrix(polygon):
    return (np.cov(polygon.T))

def mahalonobis_distance(polygon1, polygon2):
    centroid1 = calculate_centroid(polygon1)
    centroid2 = calculate_centroid(polygon2)
    
    all_points = np.vstack([polygon1, polygon2])
    
    cov_matrix = calculate_covariance_matrix(all_points)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    distance = mahalanobis(centroid1, centroid2, inv_cov_matrix)
    return distance

polygon1 = np.array([[1,1],[2,3],[3,1],[2,-1]])
polygon2 = np.array([[5,5],[6,7],[7,5],[6,3]])

distance = mahalonobis_distance(polygon1, polygon2)
print(f"Mahalanobis distance between the two polygons: {distance}")