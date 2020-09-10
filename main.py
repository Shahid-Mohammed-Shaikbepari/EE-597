"""
This file is created by Shahid Mohammed for the EE 597 home work assignment 1

"""

"""
1. Lets first calculate the X vec, x = log10(d/d0) for all the transmitter and reciever points and then create a Y vector with corresponding values

2. Take average of the all the powers in every file and then match with the X_vec values

"""
import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from copy import deepcopy




def getDistance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def fun():

    # generate_X_vector()
    eta = None
    Kref = None
    sigma2 = []
    X_vec = []
    d0 = 1 # reference distance
    transmitterLoc = pd.read_csv(
        r'D:\Users\shahi\PycharmProjects\EE597\path_loss_parameter_estimation\transmitterXY.csv', header=None)
    df_trans = pd.DataFrame(transmitterLoc)
    receiverLoc= pd.read_csv(r'D:\Users\shahi\PycharmProjects\EE597\path_loss_parameter_estimation\receiverXY.csv', header=None)
    df_receiver = pd.DataFrame(receiverLoc)
    for i in range(12):
        #read the transmitter ith location and calculate the x_vector
        x1, y1 = df_trans.iloc[i][0], df_trans.iloc[i][1]
        for j in range(8):
            x2, y2 = df_receiver.iloc[j][0], df_receiver.iloc[j][1]
            d = getDistance(x1, y1, x2, y2)
            #math.log10 imported from tutorialspoint.com
            X_vec.append([math.log10(d/d0)])

    X_scatter_values = {}
    for i in range(len(X_vec)):
        X_scatter_values[i] = X_vec[i]
    #now estimate/compute eta, kref, sigma2 for this x_vector
    column = np.ones((len(X_vec), 1), dtype=int)
    X_vec = np.concatenate((X_vec, column), 1)

    # w = (X^T*X)^-1*X^T*y
    #x_vector's transpose
    XT = X_vec.T  # X transpose

    # taken from https://pythontic.com/pandas/dataframe-binaryoperatorfunctions/dot
    mul = XT.dot(X_vec)
    inv = np.linalg.inv(mul)  # taken from tutorialspoint.com
    temp_res = inv.dot(XT)
    # print(len(temp_res[0]))

    """
    now lets create a y vector here
    """
    Y_vec = []
    noise_index_list = set()
    for num in range(7, 19):
        path = r"D:\Users\shahi\PycharmProjects\EE597\path_loss_parameter_estimation\wifiExp" + str(num)
        filePath = os.path.join(path + ".csv")
        # print(filePath)
        Prdbm = pd.read_csv(filePath, header=None)
        # print(Prdbm)
        Prdbm = pd.DataFrame(Prdbm)
        # https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe
        # https://www.w3resource.com/python-exercises/pandas/python-pandas-data-frame-exercise-57.php
        # print(len(Prdbm.index))
        # print(len(Prdbm.columns))

        for col in range(1, len(Prdbm.columns)):
            entry_count = 0
            total = 0
            for row in range(len(Prdbm.index)):
                if Prdbm.loc[row, col] == 500:
                    continue
                entry_count += 1
                total += -Prdbm.loc[row, col]
            if total == 0:
                noise_index_list.add(len(Y_vec))
                Y_vec.append(-500)
            else:
                Y_vec.append(total / entry_count)

    # print(noise_index_list)
    final_X = []
    for i in range(2):
        row = []
        for j in range(len(temp_res[0])):
            if j in noise_index_list:
                continue
            row.append(temp_res[i][j])
        final_X.append(row)

    final_Y = []
    for i in range(len(Y_vec)):
        if i in noise_index_list:
            continue
        final_Y.append([Y_vec[i]])

    # print(len(final_X), len(final_X[0]))
    #     # print("-----------------")
    #     # print(len(final_Y), len(final_Y[0]))
    final_X = np.array(final_X)
    final_Y = np.array(final_Y)
    for i in noise_index_list:
        X_scatter_values.pop(i)

    w = final_X.dot(final_Y)
    plt.scatter(X_scatter_values.values(), final_Y)
    # print(w)
    slope, intercept = w[0][0], w[1][0]
    abline(w[0][0], w[1][0])
    plt.ylabel('Prdbm')
    plt.xlabel('log(d/d0)')
    plt.show()
    # slope = -10eta
    # eta= -slope/10
    eta = -slope/10
    #y-intercept = Kref + Ptdbm
    #Kref = y_intercept - Ptdbm
    Ptdbm = -27
    Kref = intercept - Ptdbm

    sigma2 = getSigmaSquare(X_scatter_values, Y_vec, Ptdbm, Kref, eta, noise_index_list)
    print(eta, Kref, sigma2)


# taken from https://stackoverflow.com/questions/7941226/how-to-add-line-based-on-slope-and-intercept-in-matplotlib
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals)
def line(x, slope, intercept):
    return slope*x + intercept

def getSigmaSquare(X_scatter_values, Y_vect, Ptdmb, Kref, eta, noise_index):
    # sig^2 = 1/n(sigma from i=1 to n [Mmeasured(di) - Mmodel(d8i)]^2)
    #Mmeasured(di) = Prdbm/Ptdbm
    #Mmodel(di) = Kref- 10*eta*log(d/d0)
    sum = 0
    for i in range(len(Y_vect)):
        if i in noise_index:
            continue
        measured = Y_vect[i]/Ptdmb
        model = Kref - (10*eta*float(X_scatter_values[i][0]))
        sum += (measured - model)**2

    return sum/len(X_scatter_values.values())


fun()





