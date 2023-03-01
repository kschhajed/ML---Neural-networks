import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Function to scale data to avoid large values in e^(-z) of sigmoid function, returns scaled data
def scaling(dataframe):
    scaleddata = (dataframe - dataframe.mean()) / (dataframe.max() - dataframe.min())
    return scaleddata

#Function to find z and sigmoid function using provided data
def hypothesis(w, traindata):
    z = np.dot(traindata,w)
    return 1/(1+np.exp(-z))

#Function to check output on validation or test dataset
def validatingtesting(w,data,y_in):
    class_rep = np.where( hypothesis(w, data) > 0.5 , 1, 0)
    accuracy = (1- (np.average((class_rep - y_in)**2))) * 100
    return accuracy

#Method to calculate cost function
def costcal(y,h):
    cost = np.sum(- (y * np.log(h)) - ((1-y) * np.log(1 - h)))/len(h)
    return cost

def main():
    #Reading data and shuffling
    abc = pd.read_csv(r"F:\UBSem1\Intro_to_ML\project1\diabetes.csv")
    data = abc.sample(frac=1).copy()

    #Extracting original outputs in ycomplete
    data_numpy = np.asarray(data)
    ycomplete = (data_numpy[:,8]).reshape(768,1)

    #Scaling and converting to numpy array
    data_scaled = scaling(data.iloc[:,:-1])
    data_scaled = np.asarray(data_scaled)
    data_scaled =np.append(np.ones(len(data_scaled)).reshape(len(data_scaled),1) ,  data_scaled ,axis = 1)

    #Splitting data into 60% training, 20% validation and 20% testing sets
    m= int(len(data_scaled)*.6)
    m1 = int((len(data_scaled)-m)/2)

    traindata = data_scaled[0:m,:]
    validatedata = data_scaled[m:(m+m1),:]
    testdata = data_scaled[(m+m1):len(data_scaled),:]

    y_train = ycomplete[0:m]
    y_validate = ycomplete[m:(m+m1)]
    y_test = ycomplete[(m+m1):len(data_scaled)]
   
    #Initializing weights
    w = np.zeros(9).reshape(9,1)

    #Hyperparameters
    alpha = 0.0025
    epochs = 400

    h = hypothesis(w, traindata)
    valacc=[]
    trainacc=[]
    cost_train=[]
    cost_val=[]

    for i in range(epochs):
        w = w - alpha * (((h-y_train).T).dot(traindata)).T
        h = hypothesis(w, traindata)
        cost_train.append(costcal(y_train,h))
        cost_val.append(costcal(y_validate,hypothesis(w, validatedata)))
        valacc.append(validatingtesting(w,validatedata,y_validate))
        trainacc.append(validatingtesting(w,traindata,y_train))


    print("Weights after training: ", w)

    plt.figure("Accuracy vs Epochs")
    plt.plot(valacc, 'r', label = 'Validation')
    plt.plot(trainacc, 'g', label = 'Training')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc = 'upper left')
    plt.show()

    plt.figure("Cost function vs Epochs")
    plt.plot(cost_train, 'r', label = 'Training')
    plt.plot(cost_val, 'g', label = 'Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Cost function")
    plt.legend(loc = 'upper left')
    plt.show()

    print("Validation dataset accuracy:",valacc[len(valacc) - 1])

    print("Training dataset accuracy:",trainacc[len(trainacc) -1])

if __name__ == "__main__":
    main()