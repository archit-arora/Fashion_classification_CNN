'''Code built on theano and lasagne tutorial at https://github.com/ebenolson/pydata2015 and
https://github.com/Britefury/deep-learning-tutorial-pydata2016'''


import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import sklearn.cross_validation
import pickle,os,json

import theano
import theano.tensor
import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax


class Fashion_classification:
    '''A class for classifying fashion apparels using Convolutional Neural Networks'''
    
    def __init__(self,vgg16_path,labels_path,train_data_dirpath,test_data_dirpath,BATCH_SIZE=16,modify_layer='fc7',
                 learning_rate=0.001,EPOCHS=50,experiment=2):
        self.test_files=[]
        self.vgg16_params=pickle.load(open(vgg16_path))
        self.labels=json.load(open(labels_path))
        self.vgg16_net=self.build_vgg()
        self.modify_vgg(self.vgg16_net)
        if (experiment==1):
            self.X,self.y=self.load_data(train_data_dirpath,test=1)
            train_ix, test_ix = sklearn.cross_validation.train_test_split(range(len(self.y)))
            self.train_X=np.copy(self.X[train_ix])
            self.train_y=np.copy(self.y[train_ix])
            self.test_X=np.copy(self.X[test_ix])
            self.test_y=np.copy(self.y[test_ix])
            self.test_files=[self.test_files[i] for i in test_ix]
        else:
            self.train_X,self.train_y=self.load_data(train_data_dirpath)
            self.test_X,self.test_y=self.load_data(test_data_dirpath,test=1)
        self.train_X_temp,self.train_y_temp=self.get_vgg_output(self.train_X,self.train_y,modify_layer,BATCH_SIZE)
        self.test_X_temp,self.test_y_temp=self.get_vgg_output(self.test_X,self.test_y,modify_layer,BATCH_SIZE)
        self.save_vgg_output(self.train_X_temp,self.train_y_temp,train_data_dirpath,'train')
        self.save_vgg_output(self.test_X_temp,self.test_y_temp,test_data_dirpath,'test')
        self.set_optimization_parameters(learning_rate)
        self.train_modified_network(EPOCHS,BATCH_SIZE=200)
        self.classify()
        
    def build_vgg(self):
        ''' Builds the VGG-16, 16-layer model as given in the paper:
        "Very Deep Convolutional Networks for Large-Scale Image Recognition" 
        and assigns optimal weights from the ImageNet classification challenge dataset
        Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8'''

        net = {}
        net['input'] = InputLayer((None, 3, 224, 224))
        net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
        net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)
        lasagne.layers.set_all_param_values(net['prob'], self.vgg16_params['param values'])
        print "Built VGG-16 model"
        return net
            
    def modify_vgg(self,network):
        '''Replaces the last 3 fully connected layers of the VGG-16 model for optimization'''
        self.net= {}
        self.net['input'] = InputLayer((None,1,64,64))
        self.net['fc1'] = DenseLayer(self.net['input'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
        self.net['fc2'] = DenseLayer(self.net['fc1'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
        self.net['prob'] = DenseLayer(self.net['fc2'], len(self.labels),nonlinearity=softmax)
        print "Modified VGG-16 model"

    
    def get_vgg_output(self,X,y,modify_layer,BATCH_SIZE):
        '''Processes the image dataset through the original VGG-16 model to generate intermediate output
        that serves as input to the modified_vgg function'''
        X_sym = theano.tensor.tensor4()
        prediction = lasagne.layers.get_output(self.vgg16_net[modify_layer], X_sym)
        pred_fn = theano.function([X_sym], prediction)
        index_set = range(X.shape[0])
        count=0
        for chunk in self.batches(index_set,BATCH_SIZE):
            if(count==0):
                X_intermediate=pred_fn(X[chunk])
                y_intermediate=np.zeros(len(index_set))
            else: 
                X_intermediate=np.vstack((X_intermediate,pred_fn(X[chunk])))
            for k in range(len(chunk)):
                y_intermediate[BATCH_SIZE*count+k]=y[chunk[k]]
            count+=1
        X_intermediate=X_intermediate.flatten().reshape((X.shape[0],1,64,64))
        y_intermediate=y_intermediate.astype('int32')
        print "Intermediate VGG-16 output obtained"
        return X_intermediate,y_intermediate
    
    
    def batches(self,iterable, N):
            chunk = []
            for item in iterable:
                chunk.append(item)
                if len(chunk) == N:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk  
                
    def save_vgg_output(self,X,y,path,type_str):
        '''Saves the intermediate output generated from passing the image dataset through the original VGG-16 model'''
        np.save(path+'/'+type_str,X)
        np.save(path+'/'+type_str,y)
        print "saved intermediate VGG-16 output"

    def set_optimization_parameters(self,learning_rate):
        '''Defines the optimization parameters(loss function, update rule) for the modified VGG-16 network'''
        self.X_sym = theano.tensor.tensor4()
        self.y_sym = theano.tensor.ivector()
        self.prediction = lasagne.layers.get_output(self.net['prob'], self.X_sym)
        self.loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.y_sym)
        self.loss = self.loss.mean()      
        self.params = lasagne.layers.get_all_params(self.net['prob'], trainable=True)
        self.grad = theano.tensor.grad(self.loss, self.params)
        self.updates = lasagne.updates.adam(self.grad, self.params, learning_rate=learning_rate)
        self.train_fn = theano.function([self.X_sym, self.y_sym], self.loss, updates=self.updates)
        self.pred_fn = theano.function([self.X_sym], self.prediction)
        print "optimization parameters set"
    
    def train_modified_network(self,EPOCHS,BATCH_SIZE):
        '''Uses the modified VGG-16 model to learn from training data'''
        for epoch in range(EPOCHS):
            train_loss=0.
            for batch in range(25):
                index=np.random.choice(len(self.train_X_temp),BATCH_SIZE,replace=False)
                train_loss += self.train_fn(self.train_X_temp[index],self.train_y_temp[index])
            print epoch,train_loss/25
        print "trained modified VGG-16 network"
            
    def classify(self,BATCH_SIZE=800):
        '''Predicts fashion classification labels for test data'''
        index = range(len(self.test_y_temp))
        for chunk in self.batches(index, BATCH_SIZE):
            self.y_pred = self.pred_fn(self.test_X_temp[chunk])
        self.y_pred= np.argmax(self.y_pred,axis=1)
        print "classified test data"
    
    def calculate_accuracy(self):
        '''Calculates prediction accuracy for test dataset'''
        self.accuracy= sum(self.y_pred==self.test_y_temp)/float(len(self.test_y_temp))
   
    def load_data(self,data_dirpath,test=0):
        '''Loads images from disk for training and test datasets and processes them for input 
        to VGG-16 model'''
        X = []
        y = []
        for category in self.labels.keys():
            for image_name in os.listdir(data_dirpath+'/'+str(category)):
                if '.jpg' in image_name:
                    img_path=data_dirpath+'/'+str(category)+'/'+str(image_name)
                    _, im = self.prep_image(img_path)
                    X.append(im)
                    y.append(category)
                    if(test==1):
                        self.test_files.append(img_path)
        X = np.concatenate(X)
        y = np.array(y).astype('int32')
        print "loaded data"
        return X,y
        
    def prep_image(self,fn, ext='jpg'):
        '''Process the images to feed as input for VGG-16 model'''
        from lasagne.utils import floatX
        IMAGE_MEAN = self.vgg16_params['mean value'][:, np.newaxis, np.newaxis]
        im = plt.imread(fn, ext)

        # Resize so smallest dim = 256, preserving aspect ratio
        h, w, _ = im.shape
        if h < w:
            im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
        else:
            im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)
    
        # Central crop to 224x224
        h, w, _ = im.shape
        im = im[h//2-112:h//2+112, w//2-112:w//2+112]
        
        rawim = np.copy(im).astype('uint8')
        
        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
        
        # discard alpha channel if present
        im = im[:3]
    
        # Convert to BGR
        im = im[::-1, :, :]
    
        im = im - IMAGE_MEAN
        return rawim, floatX(im[np.newaxis])
    
    def show_results(self,grid_size=(10,5),im_range=(0,50)):
        '''Displays the predictions labels for the images in test dataset'''
        plt.figure(figsize=(12, 24))
        lb,ub=im_range
        j=lb
        for i in range(ub-lb):
            plt.subplot(grid_size[0],grid_size[1],i+1)
            test_image=plt.imread(self.test_files[j])
            plt.imshow(test_image)
            true =self.test_y[j]
            pred = self.y_pred[j]
            color = 'green' if true == pred else 'red'
            plt.title(self.labels[str(pred)], color=color, bbox=dict(facecolor='white', alpha=1))
            j+=1

            plt.axis('off')
            
if __name__=='__main__':
        #Experiment 1
        vgg16_path='vgg16.pkl'
        labels_path='LABELS.txt'
        train_data_dirpath='/Datasets/images_macys_original'
        test_data_dirpath=' '
        f1=Fashion_classification(vgg16_path,labels_path,train_data_dirpath,test_data_dirpath,experiment=1)
        #Experiment 2
        vgg16_path='vgg16.pkl'
        labels_path='LABELS.txt'
        train_data_dirpath='/Datasets/images_macys_augmented'
        test_data_dirpath='/Datasets/test_data'
        f2=Fashion_classification(vgg16_path,labels_path,train_data_dirpath,test_data_dirpath)
