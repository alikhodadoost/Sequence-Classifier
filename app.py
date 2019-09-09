import random
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def generateData(numSequences=100000,fixedLen=True,maxLen=50):
    '''Generates Data
    
    Parameters:
        fixedLen (bool): generete fixed length sequences or not
        maxLen (int): maximum sequence length, if fixedLen is True length of sequences equals maxLen
    
    Returns:
        list, list
            list of sequences and list of targets 

    '''
    sequences =[]
    targets=[]
    if fixedLen:
        for i in range(numSequences):
            sequences.append([str(random.randint(0,1)) for j in range(maxLen)])

        for val in sequences:
            tmp=[int(v) for v in val]
            if sum(tmp)%2==0:
                targets.append(['1','0'])
            else:
                targets.append(['0','1'])
    else:
        for i in range(numSequences):
            sequences.append([str(random.randint(0,1)) for j in range(random.randint(1,maxLen))])

        for val in sequences:
            tmp=[int(v) for v in val]
            if sum(tmp)%2==0:
                targets.append(['1','0'])
            else:
                targets.append(['0','1'])
    
    return sequences,targets

def writeToFile(sequences,targets,outputfile='input.txt'):
    '''Writes the data to file
    
    Parameters:
        sequences (list): sequences
        targets (list): corresponding targets of sequences
    '''
    assert len(sequences)==len(targets)
    
    for i in range(len(sequences)):
        if i%500==0:
            print(i)
        with open(outputfile,'a+') as f:
            f.write(''.join(sequences[i])+','+''.join(targets[i])+'\n')

def readData(file='input.txt',delimiter=',',padSequences=False,maxSeqLen=50):
    '''Read data from file
    
    Parameters:
        file (str): the file containing the data
        delimiter (str): delimiter(seperator) of source and target in data lines
        padSequences (bool) : pad the sequences to have the same length

    Returns:
        list, list
            list of sequences and list of targets
    '''
    with open(file,'r') as f:
        lines = f.readlines()
    lines = [l.replace('\n','').split(delimiter) for l in lines]
    
    #list one hot encoded targets
    targets =[[int(val) for val in list(l[1])] for l in lines]

    #One Hot Encode Sequences
    if padSequences==True:
        vals=[]
        for l in lines:
            vals.append(list(l[0]))
        paddedVals= tf.keras.preprocessing.sequence.pad_sequences(vals,maxlen=maxSeqLen,dtype='str',padding='post',value='P')
        
        sequences=[]
        for p in paddedVals:
            seq=[]
            for val in p:
                if val=='0':
                    seq.append([1,0,0])
                elif val=='1':
                    seq.append([0,1,0])
                else:
                    seq.append([0,0,1])
            sequences.append(seq)

    else:
        sequences=[]
        for l in lines:
            seq=[]
            vals=list(l[0])
            for val in vals:
                if val=='0':
                    seq.append([1,0])
                else:
                    seq.append([0,1])
            sequences.append(seq)
    
    #Assertion to have sequences and targets of the same length
    assert len(sequences) == len(targets)

    return sequences,targets

def createModel(lstmUnits,inputShape):
    '''Dfinition of model

    Parameters:
        lstmUnits (int): Number of lstm units
        inputShape (tuple): Input shape of the model

    Returns:
        model
    '''

    inputLayer=tf.keras.layers.Input(shape=inputShape)
    lstm = tf.keras.layers.LSTM(lstmUnits)(inputLayer)
    outLayer = tf.keras.layers.Dense(2,activation='sigmoid')(lstm)
    
    opt = tf.keras.optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

    model = tf.keras.Model(inputs=inputLayer,outputs = outLayer)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acc'])

    return model

def evalModel(model,x,y):
    '''Evaluates the model using the given test samples
    Parameters:
        model (Model): given model object
        x (numpy.ndarray): Input samples
        y (numpy.ndarray): Expected targets
    '''
    for i in range(x.shape[0]):
        print(i)
        X = np.array([x[i]])
        Y = np.array([y[i]])
        model.evaluate(X,Y,batch_size=1)




if __name__=='__main__':
    random.seed(5)
    
    # s,t = generateData(fixedLen=False)
    # writeToFile(s,t)
    s,t = readData(file='input2.txt',padSequences=True,maxSeqLen=50)

    if len(s[0][0])==3:
        INPUTSHAPE = (len(s),50,3)
    elif len(s[0][0])==2:
        INPUTSHAPE=(len(s),50,2)
    
    s = np.array(s).reshape(INPUTSHAPE)
    t = np.array(t)

    xTrain,xTest,yTrain,yTest = train_test_split(s,t,test_size=0.25)

    LSTMUNITS = 128
    EPOCHS= 50
    
    model = createModel(lstmUnits=LSTMUNITS,inputShape=INPUTSHAPE[1:])
    model.summary()
    for _ in range(50):
        for i in range(xTrain.shape[0]):
            print(i)
            X = np.array([xTrain[i]])
            Y = np.array([yTrain[i]])
            model.fit(X,Y,epochs=1,verbose=2) 
    model.save('myModel.h5') 
    
    model = tf.keras.models.load_model('myModel.h5')

    evalModel(model,xTest,yTest)
    
