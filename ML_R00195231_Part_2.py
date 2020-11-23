from ML_R00195231_Part_1 import KNN_Regression
import numpy as np


class KMeans_Cluster:

    ## This methid is used for generate centroids
    def generate_centroids(self, data, k):
        n_rows = data.shape[0]                      ## It gives the dimension of array
        indices = np.random.choice(n_rows, size=k, replace=True)   ## choose randomly k number of index from data sets
        centroids_list = data[indices, :]
        return centroids_list                       ## return centroids list that are choosen randomly

    ## This method is used for Assign Centroids
    def assign_centroids(self, data, centroids):
        temp = []
        for i in range(len(centroids)):
            centroids_dis = reg.calculate_distances(data, centroids[i])     ## calculate distance from each point of data set to each centroids
            temp.append(centroids_dis)
        dist = np.array(temp)
        trans = np.transpose(dist)      ## change the dimension  i.e (3,10000) is converted into (10000,3)
        centroid_index = np.argmin(trans, axis=1)           ## fetch min along with row
        return centroid_index

    ## This method is used for moving centroids
    def move_centroid(self, data, centroid_index, centroids_list):
        return np.array([data[centroid_index == i].mean(axis=0) for i in range(centroids_list.shape[0])])  ## for each data points , closest centroids index is stored in array and take mean values that gives new centroids values


    ## This method is used for calculating distortion cost
    def calculate_cost(self, data, centroid_index, centroids_list):
        return np.divide(np.array(
            [np.sum(reg.calculate_distances(data[centroid_index == i], centroids_list[i]) ** 2) for i in
             range(len(centroids_list))]).sum(), len(data))                 ## for calculating cost, distcalculate method is used and squaring it for each centroids , gives cost value

    ## This method is used running code multiple times and gives list of best centroids fro clusters
    def restart_KMeans(self, data, number_of_centroids, number_of_iterations, number_of_restarts):
        sol = []
        centroids = []
        global centroid_index, sol_np, centroids_np
        for k in range(number_of_restarts):
            centroid_list = cluster.generate_centroids(data, number_of_centroids)  ## centroids are generating randomly
            for i in range(number_of_iterations):
                centroid_index = self.assign_centroids(data, centroid_list)         ## centroids are assigning for each iteration
                centroid_list = self.move_centroid(data, centroid_index, centroid_list)  ## centroids are moving for each iteration
            sol.append(self.calculate_cost(data, centroid_index, centroid_list))
            centroids.append(centroid_list)             ## stored centroids for each iterations
        res = np.min(sol)                               ## take min cost among all cost , gives best soltuion
        best_centroids_index = sol.index(min(sol))      ## take same inedx of min cost
        return res, centroids[best_centroids_index]     ## fetch same centroids for which min cost is getting


if __name__ == '__main__':
    import time

    print('##########################')
    print('***************************************************')
    print(" The Output array has two values ")
    print("1. Best distortion cost value ")
    print("2. Array of best converged centroids ")
    start_time = time.time()
    Value_of_K = 3
    reg = KNN_Regression(Value_of_K)
    cluster = KMeans_Cluster()
    data = reg.read_data('Data.csv')
    print("Output array is :  ", cluster.restart_KMeans(data, Value_of_K, 10, 10))
    print("time taken in algo : ", time.time() - start_time)
    print('##########################')
    print('***************************************************')

