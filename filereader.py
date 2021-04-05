import csv
import numpy as np

#read the input for the NN as csv file
def returnInput():
    #calculate rows and 
    with open('input.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        numRows = 0
        numCols = 0
        for row in csv_reader:
            if(numRows == 0):
                for val in row:
                    numCols += 1
            if(row[0] == "END"):
                break
            numRows += 1
        print(numRows, "|", numCols)
        x_matrix = np.empty((numRows, numCols))
        y_matrix = np.empty((1, numCols))
    
    with open('input.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        currRow = 0
        for row in csv_reader:
            print(currRow)
            currCol = 0
            if(currRow < numRows):
                for val in row:
                    x_matrix[currRow, currCol] = float(val)
                    currCol += 1
            elif(currRow > numRows):
                for val in row:
                    y_matrix[0, currCol] = float(val)
                    currCol += 1
            currRow += 1
        print(x_matrix)
        print(y_matrix)
        return (x_matrix, y_matrix)

        
