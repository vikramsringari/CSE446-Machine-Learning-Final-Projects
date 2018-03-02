import numpy as np
import keras
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt

#takes in a File and a dev value if your are using development set
def parse(path_x,num_csv,path_y=False):
    X = []
    Y = []
    if path_y:
        with open(path_y) as f:
                content = f.readlines()
                for line in range(int(len(content))):
                     Y.append(float(content[line]))
                     
    for i in range(num_csv):
        k = i+1     
        fil = str(k)+".csv"
        with open(path_x+fil) as f:
            content = f.readlines()
            feature_num = len((content[0].split(",")))
            feature_values = content[0].split(",")
            x = [float(feature_values[i]) for i in range(len(feature_values))]
            X.append(x)
        
    X = np.matrix(X)
    Y = np.asarray(Y)
    return X, Y, feature_num # returns X data and Y data as well as a the size of the feature vector



'''
Constructs Deep Neural Net model architecture
returns model, and X, and Y data
'''
def constructModel():
    
    X, Y, features = parse("competition-datasets-utf-8/projectgroup8_miri-playlist-dataset/train/",10503,path_y="competition-datasets-utf-8/projectgroup8_miri-playlist-dataset/train-y")
    model = Sequential()
    model.add(Dense(16, input_dim=features, activation='relu'))# sets 22 as the number of nodes in the first layer. Sets the input dimension to feature vector length
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(1))# set the final activation function to softmax
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return model, X, Y

'''
fits and trains model
'''
def main():
    
    model, X_train, Y_train = constructModel()
    #tuning learning rate hyperparameters 
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2000, min_lr=0.001)
    #tuning epochs based on loss 
    early_stopping = EarlyStopping(monitor='val_loss', patience=2000, verbose=None, mode='auto')
    history = model.fit(X_train, Y_train, epochs=10000, batch_size=1000, validation_split=0.2, callbacks=[reduce_lr, early_stopping])
    scores = model.evaluate(X_train, Y_train)# score the model
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1])) # training accuracy
    prediction = model.predict(X_train, batch_size=1000) # returns the predicted y values 
    print(prediction)
    X_test, Y_test, features = parse("competition-datasets-test-no-test-y/projectgroup8_miri-playlist-dataset/test/",2626) #only X_test used
    prediction = model.predict(X_test, batch_size=1000) # returns the predicted y values (for test)
    print(prediction)
    print("Writing")
    with open("40.8.test-yhat", "w") as f:
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
    ax.set_title('Training Loss Miri Playlist Dataset')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    
    fig.savefig("miri_loss.png") # training loss plot
    
    
if __name__ == "__main__":
    main()

