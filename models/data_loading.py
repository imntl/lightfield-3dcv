import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

class LightFieldDataset:
    '''
    Dataset class for the Lightfield Dataset.
    '''
    
    def __init__(self, sort = 'training', data_kind = 'grid', root_dir = './data/', transform=None):
        '''
        Args:
            sort (string or list): What sort of data you want to load. Valid input: 'training', 'test', 'stratified', 'additional'.
            data_kind (string): What structure the data should have. Valid input: 'grid', 'hstack', 'vstack', 'stack', 'cross'.
            root_dir (string, optional): the working directory with subdirectories 'training', 'test', etc.
            transform (callable, optional): Optional transform to be applied the data.
        '''
        
        self.data_kind = data_kind
        self.root_dir = root_dir
        self.transform = transform
        self.directory_names = []
        self.images_paths = []
        self.stack_index = []
        
        if type(sort) == str:
            self.images_paths.append(os.path.join(root_dir, sort))
        else:
            for directory in sort:
                self.images_paths.append(os.path.join(root_dir, directory))
        
        # this block just stores the path of each individual scene (=grid)
        # iterates over all entered values in 'sort'.
        for images_path in self.images_paths:
            # safe all the scene names (directories) and all image names of each scene
            # iterate over all directories in images path. Each directory includes the images of *one* scene
            for filename in os.listdir(images_path):
                directory_path = os.path.join(images_path, filename)
                if os.path.isdir(directory_path):
                    if self.data_kind in ['grid', 'cross']:
                        self.directory_names.append(directory_path)
                    elif self.data_kind == 'stack':
                        # for data_kind stack we have 18 stacks per scene, 9 hstacks and 9 vstacks.
                        for i in range(18):
                            self.directory_names.append(directory_path)
                            self.stack_index.append(i)
                    else:
                        # for data_kind hstack or vstack, we have 9 stacks per scene, so we need to call each scene
                        # 9 times in __getitem__ and also add a stack_index to know whether images are in the same
                        # stack.
                        for i in range(9):
                            self.directory_names.append(directory_path)
                            self.stack_index.append(i)

    def __len__(self):
        '''
        Returns the lenght of the dataset (number of grid's/ hstack's/ vstack's/ stack's/ crosses)
        '''
        return len(self.directory_names)
    
    def __getitem__(self, index):
        '''
        Returns a whole grid/ vstack/ hstack/ stack/ cross
        '''
        
        final_data = []
        
        if torch.is_tensor(index):
            index = index.tolist()
        
        # iterate over all files for a fixed scene (which is one directory name)
        for imagename in sorted(os.listdir(self.directory_names[index])):
            # read all images
            if imagename[0:5] == 'input':
                if self.data_kind == 'grid':
                    img = cv2.imread(os.path.join(self.directory_names[index], imagename))
                    # To resize all images to a specific shape, uncomment: img = cv2.resize(img, shape)
                    # append each color channel seperately
                    for color in range(3):
                        final_data.append(img[:, :, color]/255)
                else:
                    # get the number of the image and omit the first zero for calculation reasons
                    if imagename[10:12].isdigit():
                        if imagename[10] == '0':
                            image_number = int(imagename[11])
                        else:
                            image_number = int(imagename[10:12])                                
                            
                        if self.data_kind == 'hstack' and np.floor_divide(image_number, 9) == self.stack_index[index]:
                            img = cv2.imread(os.path.join(self.directory_names[index], imagename))
                            for color in range(3):
                                final_data.append(img[:, :, color]/255)
                                
                        if self.data_kind == 'vstack' and image_number % 9 == self.stack_index[index]:
                            img = cv2.imread(os.path.join(self.directory_names[index], imagename))
                            for color in range(3):
                                final_data.append(img[:, :, color]/255)
                                
                        if self.data_kind == 'stack':
                            if self.stack_index[index] <= 8 and np.floor_divide(image_number, 9) == self.stack_index[index]:
                                img = cv2.imread(os.path.join(self.directory_names[index], imagename))
                                for color in range(3):
                                    final_data.append(img[:, :, color]/255)
                            if self.stack_index[index] > 8:
                                custom_stack_index = self.stack_index[index] - 9
                                if image_number % 9 == custom_stack_index:
                                    img = cv2.imread(os.path.join(self.directory_names[index], imagename))
                                    for color in range(3):
                                        final_data.append(img[:, :, color]/255)
                        
                        if self.data_kind == 'cross' and ((np.floor_divide(image_number, 9) == 4) or 
                                                     (image_number != 40 and image_number % 9 == 4)):
                            img = cv2.imread(os.path.join(self.directory_names[index], imagename))
                            for color in range(3):
                                final_data.append(img[:, :, color]/255)
        
        # in the 'cross' case the images got appended from vertical to horizontal, but we swap it the other way around
        if self.data_kind == 'cross':
            final_data = final_data[4:13] + final_data[0:4] + final_data[13:]
            
        final_data = np.array(final_data)
                    
        if self.transform:
            final_data = self.transform(final_data)
        
        return final_data
    
'''
Code, welcher eine Batch an Daten annimmt und diese dann mit MatPlotLib anzeigt.
'''

def lightPlotterGrid(dataloader, iterations = 1, save = False):
    for batch in dataloader:
        for data in batch:
            # Iterator to show just interations of images
            iterations -= 1
            if iterations < 0: return            
            
            # Calculate number of images and number of rows for grid
            n_img = int(data.size()[0]/3)
            n_rows = int(np.sqrt(n_img))
            
            # Plotting setup
            fig = plt.figure(figsize=(30,30))
            axs = fig.subplots(n_rows,n_rows)
            for i in range(n_img):
                # Concatenate color channels of images
                img = np.zeros((data.size()[1],data.size()[2],3))
                for color in range(3):
                    img[:,:,color] = data[(i*3)+color,:,:].numpy()
                # Plot
                axs[i//n_rows,i%n_rows].get_xaxis().set_visible(False)
                axs[i//n_rows,i%n_rows].get_yaxis().set_visible(False)
                axs[i//n_rows,i%n_rows].imshow(img)
            if save: fig.savefig(f"grid{iterations}.png")
    
