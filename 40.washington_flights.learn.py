import numpy as np
import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping  
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

def __deepcopy__(self):
    val = self
    return val

'''
parses through training files to return X and Y data
also returns length of feature vector
Takes x path, number of csv files for the x training, and conditional y path (only for training data)
'''
def parse(path_x,num_csv,path_y=False):
    X = []
    Y = []
    if path_y:
        with open(path_y) as f:
                content = f.readlines()
                for line in range(int(len(content))):
                     Y.append(float(content[line]))
                 
    for i in range(num_csv):   
        fil = str(i)+".csv"
        with open(path_x+fil) as f:
            content = f.readlines()
            feature_num = len((content[0].split(",")))
            feature_values = content[0].split(",")
            c = []
            x = []
            for i in range(len(feature_values)):
                if i != 2 and i != 3 and i != 4 :
                    x.append(float(feature_values[i]))
                elif feature_values[i]  in c:
                    x.append(float(c.index(feature_values[i])))
                else:
                    x.append(float(len(c) + 1))
            X.append(x)
        
    X = np.matrix(X)
    Y = np.asarray(Y)
    return X, Y, feature_num # returns X data and Y data as well as a the size of the feature vector

'''
Constructs Deep Neural Net model architecture
returns model, and X, and Y data
'''
def constructModel():
    X, Y, features = parse("competition-datasets-utf-8/projectgroup25_washington-flights-dataset/train/",10044,path_y="competition-datasets-utf-8/projectgroup25_washington-flights-dataset/train-y")
    # initialize model
    model = Sequential()
    # set input layer to feature vector length
    model.add(Dense(256, input_dim=features, activation='relu'))
    model.add(Dropout(0.05)) 
    # add hidden layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.05))
    # returns one output in last layer
    model.add(Dense(1))
    # comile the model using adam optimizer and categorical crossentropy (logloss)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return model, X, Y


    

'''
fits and trains model
'''
def main():
    
    
    model, X_train, Y_train = constructModel()
    
    #tuning learning rate hyperparameters 
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2500, min_lr=0.001)
    #tuning epochs based on loss 
    early_stopping = EarlyStopping(monitor='val_loss', patience=2500, verbose=None, mode='auto')
                              
    #train our model architecture on X and Y training
    history = model.fit(X_train, Y_train, epochs=10000, batch_size=1000, validation_split=0.2, callbacks=[reduce_lr, early_stopping])

    scores = model.evaluate(X_train, Y_train)# score the model
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1])) # training accuracy
    prediction = model.predict(X_train, batch_size=1000) # returns the predicted y values 
    print(prediction)
    #tuneHP(model, X_train, Y_train)
    X_test, Y_test, features = parse("competition-datasets-test-no-test-y/projectgroup25_washington-flights-dataset/test/",1116) #only X_test used
    prediction = model.predict(X_test, batch_size=1000) # returns the predicted y values (for test)
    print(prediction)
    print("Writing")
    with open("40.25.test-yhat", "w") as f:
        for line in prediction:
            f.write(str(int(line)) + "\n")
    print("Write Complete")
    
    '''
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(history.history['acc'])
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    fig.savefig("accuracy.png") # training accurcay plot
    '''
    # graphing loss
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(history.history['loss'], label='training loss')
    ax.plot(history.history['val_loss'], label = 'validation loss')
    ax.legend()
    ax.set_title('Training Loss for Washington Flights')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    fig.savefig("loss.png") # training loss plot

    
if __name__ == "__main__":
    main()
