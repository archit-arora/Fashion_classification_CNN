#File for augmenting image dataset downloaded from Macys website 

import os,json
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from PIL import ImageEnhance, Image

dir_path=os.getcwd()
labels_path=dir_path+'LABELS.txt'

labels=json.load(open(labels_path))
categories=labels.keys()

for i in categories:
    read_path=dir_path+'/images-original/'+i
    save_path=dir_path+'/images-modified'
    os.chdir(read_path)
    print save_path
    print os.listdir(read_path)
    if i not in os.listdir(save_path):
        save_path=save_path+'/'+i
        os.mkdir(save_path)
    else:
        save_path=save_path+'/'+i
    for image_name in os.listdir(read_path)[:200]:
        if (image_name.startswith('.')):
            continue
        options={0:'pass',1:'noise',2:'rotate',3:'contrast',4:'jitter',5:'crop',6:'scale'}
        image=plt.imread(read_path+'/'+image_name)
        
        #Add background noise to image with probability 0.5
        option= np.random.randint(0,2)
        if option==1:
            color={0:'noise',1:'solid'}
            color_choice=np.random.randint(0,2)
            if color_choice==0:
                image_transformed=image.flatten()
                image_transformed[image_transformed>244]=np.random.randint(0,255,len(image_transformed[image_transformed>244]))
                image=image_transformed.reshape(*image.shape)
            else:
                image_transformed=image.flatten()
                image_transformed[image_transformed>244]=np.random.randint(0,255)
                image=image_transformed.reshape(*image.shape)
        
        #rotate image between -10 to 10 degrees with 0.5 probability
        option= np.random.randint(0,2)
        if option==1:
            angle=np.random.randint(-10,10)
            image=skimage.transform.rotate(image,angle)
        

        #add random jitter to image with 0.2 probability
        option= np.random.randint(0,5)
        if option==1:
            image=skimage.util.random_noise(image)
        
        #crop image from sides with 0.5 probability
        option= np.random.randint(0,2)
        if option==1: 
            axis1_crop=tuple(np.random.randint(0,100,2))
            axis2_crop=tuple(np.random.randint(0,40,2))
            image=skimage.util.crop(image,(axis1_crop,axis2_crop,(0,0)))

        #scale with 0.25 probability
        option= np.random.randint(0,4)
        if option==1:                                  
            scale=0.5+np.random.random()*1.5
            image=skimage.transform.rescale(image, scale)
        
        fig = plt.imshow(image)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(save_path+'/'+image_name,bbox_inches='tight',pad_inches=0) #save image
        
        #change contrast with 0.5 probability
        option= np.random.randint(0,2)
        if option==1:
            image=Image.open(save_path+'/'+image_name)
            contrast=ImageEnhance.Contrast(image)
            contrast_factor=0.5+np.random.random()*1.5
            image_transformed=contrast.enhance(contrast_factor)
            plt.imshow(image_transformed)
            plt.savefig(save_path+'/'+image_name, bbox_inches='tight',pad_inches=0) #save image


