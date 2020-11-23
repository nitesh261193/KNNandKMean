import numpy as np
from sklearn.preprocessing import normalize


class KNN_Regression:
    def __init__(self, k):
        self.col = None
        self.k = k

    ## This method is used for read data from CSV file 
    def read_data(self, Path):
        data = np.genfromtxt(Path, dtype=float, delimiter=',')
        return data

    ## this method is used for calculate distance from query instance to each train data points
    def calculate_distances(self, train_data, query):
        self.col = len(train_data[0])
        result = train_data[:, :self.col - 1] - query[:self.col - 1]
        sqre = np.square(result)  ## squaring values
        dist_sum = np.sum(sqre, axis=1)  ## summing up along the rows
        dist = np.sqrt(dist_sum)  ## sqaure root
        return dist

    ## this method - to predict the traget value for new instance
    def predict(self, traindata, query):
        dist = self.calculate_distances(traindata, query)  ## calculate distance for each data points to one instance
        target_value = traindata[:, self.col - 1]
        dict = np.hstack((dist.reshape(len(dist), 1), target_value.reshape(len(target_value),
                                                                           1)))  ## Distance and target value is mapped in one array
        sort_dist = dict[np.argsort(dict[:, 0])]   ## sort on the basis of first column
        shortest_dist = sort_dist[0:self.k]             ## slicing
        div = np.divide(1, shortest_dist[:, 0])
        inv_squr = np.square(div)                       ## inverse wt distance logic starts here
        mul_wt = np.multiply(inv_squr, shortest_dist[:, 1])  ## Multiply square values to target values
        value = np.sum(mul_wt)                  ##  summing up multiply wt
        denom = np.sum(inv_squr)
        value_final = np.divide(value, denom)
        return value_final

    ## This method is used for calculating r2 value
    def calculate_r2(self, solution, testdata):
        values = solution - testdata[:, self.col - 1]       ## Predicted target values subtracted with test data's target values
        sqre = np.square(values)                            ## sqaure values
        sum_residual = np.sum(sqre)
        mean_target_value = np.mean(testdata[:, self.col - 1])    ## Means of target values of testdata
        values_target = mean_target_value - testdata[:, self.col - 1]  ## substarcted mean and target value of testdata
        sqre_target = np.square(values_target)
        sum_sqrs = np.sum(sqre_target)
        r2 = 1 - np.divide(sum_residual, sum_sqrs)          ## divide  sum of square residuals and tolal sum of squares
        return r2

    ### Part 1 B
    ## this method is designed for normalise data
    def normalizrion_data(self, train_data, test_data):
        train_normalised = normalize(train_data[:, :self.col - 1], axis=0, norm='l1')  ## Normalize data using MaxMinScaler Tech
        test_normalised = normalize(test_data[:, :self.col - 1], axis=0, norm='l1')

        train_y = train_data[:, -1]                                 ## fetch target values
        test_y = test_data[:, -1]

        train = np.column_stack((train_normalised, train_y))        ## merge target values to normalize data
        test = np.column_stack((test_normalised, test_y))

        np.savetxt("train_nrm.csv", train, delimiter=",")           ## save CSV file in same location with different name
        np.savetxt("test_nrm.csv", test, delimiter=",")


if __name__ == '__main__':
    import time

    print('##########################')
    print('***************************************************')
    start_time = time.time()
    reg = KNN_Regression(3)
    solution = []
    traindata = reg.read_data('trainingData.csv')
    testdata = reg.read_data('testData.csv')
    for i in range(0, len(testdata)):
        solution.append(reg.predict(traindata, testdata[i]))
    r2 = reg.calculate_r2(solution, testdata)
    print("Calculated r2 value is : ", r2)
    print("time taken in algo : ", time.time() - start_time)
    print('##########################')
    print('***************************************************')

    ## After normalisation  ,

    print('##########################')
    print('***************************************************')
    start_time = time.time()
    reg.normalizrion_data(traindata, testdata)
    print("After Normalization ")
    solution = []
    traindata = reg.read_data('train_nrm.csv')
    testdata = reg.read_data('test_nrm.csv')
    for i in range(0, len(testdata)):
        solution.append(reg.predict(traindata, testdata[i]))
    r2 = reg.calculate_r2(solution, testdata)
    print("Calculated r2 value is : ", r2)
    print("time taken in algo : ", time.time() - start_time)
    print('##########################')
    print('***************************************************')
