# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""
#%%
import numpy as np
from scipy import optimize
from math import *
#%%
def Squared_Expo(x1,x2,y1,y2,a,h):
    _x1,_x2=np.meshgrid(x1,x2)
    _y1,_y2=np.meshgrid(y1,y2)
    return a**2*np.exp(-((_x1-_x2)**2+(_y1-_y2)**2)/(2*h**2))
#%%
def Ornstein_U(x1,x2,y1,y2,a,h):
    _x1,_x2=np.meshgrid(x1,x2)
    _y1,_y2=np.meshgrid(y1,y2)
    return a**2*np.exp(-np.sqrt((_x1-_x2)**2+(_y1-_y2)**2)/h)
#%%
def Periodic(x1,x2,y1,y2,a,h):
    _x1,_x2=np.meshgrid(x1,x2)
    _y1,_y2=np.meshgrid(y1,y2)
    return a**2*np.exp(-2*np.sin(np.sqrt((_x1-_x2)**2+(_y1-_y2)**2)/2)**2/h**2)
#%%
def GPR_Kernel (a,h,sig_data,K=Squared_Expo,width=9,badpix=np.array([[0],[0]]),close_badpix=None):
    x=np.linspace(badpix[0]-(width-1)/2,badpix[0]+(width-1)/2,width)
    y=np.linspace(badpix[1]-(width-1)/2,badpix[1]+(width-1)/2,width)
    x,y=np.meshgrid(x,y)
    # Reshape into an 1D vector of coordinates
    x=np.reshape(x,width**2)
    y=np.reshape(y,width**2)
    all_badpix=badpix.copy()
    if close_badpix is not None:
        all_badpix=np.append(badpix,close_badpix,axis=1)
    # Check each point (x,y) for bad pixels to get the boolean array of good pixels
    good_pix=~(np.vstack((y,x)).T[:,None]==all_badpix.T).all(axis=-1).any(axis=-1)
    Cov_data=np.identity(x[good_pix].size)*sig_data**2    
    Kinv=np.linalg.inv(K(x[good_pix],x[good_pix],y[good_pix],y[good_pix],a,h)+Cov_data)
    Kernel=np.zeros(width**2)
    Kernel[good_pix]=np.dot(K(x[good_pix],badpix[0,0],y[good_pix],badpix[1,0],a,h),Kinv)[0]
    # Normalization
    Kernel/=np.sum(Kernel)
    # Reshape back to 2D
    Kernel=np.reshape(Kernel,[width,width])
    return Kernel
#%%
def GPR_fix(a,h,image,BP,sig_data=1,K=Squared_Expo,width=9,fill=1.0):
    # Supports images with bad pixels labeled as NaN
    if BP is "asnan":
        mask=np.isnan(image)
        # Convert Mask to indices of bad pixels
        BP_indices=np.asarray(np.nonzero(mask))
        # Replace NaN with a finite value for the training process
        image[mask]=1
    # Supports bad pixels indicated by a boolean mask
    elif BP.dtype == bool:
        BP_indices=np.asarray(np.nonzero(BP))
    else:
        BP_indices=BP
    shape=image.shape
    Nbad=BP_indices.shape[1]
    fixed_pix=np.zeros(Nbad)
    GPR_fixed_images=image.copy()
    residual=np.zeros(Nbad)
    h_width=int((width-1)/2)
    # Precompute the kernel for the case of no additional bad pixels
    perfect_kernel=GPR_Kernel(a,h,sig_data,K=K,width=width)
    for i in range (Nbad):
        # Locate close bad pixels
        #  isclose=np.logical_and(np.abs(BP_indices[0]-BP_indices[0,i])<=h_width,np.abs(BP_indices[1]-BP_indices[1,i])<=h_width)
        isclose=np.all(np.abs((BP_indices.T-BP_indices[:,i].T).T)<=h_width,axis=0)
        # Remove the central pixel itself
        #close_not_central=np.logical_and(isclose,np.logical_or(BP_indices[0]!=BP_indices[0,i],BP_indices[1]!=BP_indices[1,i]))
        close_not_central=np.logical_and(isclose,np.any(BP_indices!=BP_indices[:,i,None],axis=0))
        close_bad=BP_indices[:,close_not_central]
        # Convert to coordinates with central pixel at (0,0)
        close_bad=(close_bad.T-BP_indices[:,i].T).T
        # Case 1: Far from edges
        if h_width <= BP_indices[0,i] < (shape[0]-h_width) and h_width <= BP_indices[1,i] < (shape[1]-h_width):
            # Construct the matrix of pixel values used in the convolution
            img=image[(BP_indices[0,i]-h_width):(BP_indices[0,i]+h_width+1), \
                            (BP_indices[1,i]-h_width):(BP_indices[1,i]+h_width+1)]
            if close_bad.size==0:
                kernel=perfect_kernel
            else:
                kernel=GPR_Kernel(a,h,sig_data,K=K,width=width,close_badpix=close_bad)
            fixed_pix[i]=np.sum(kernel*img)
        # Case 2: Top Edge
        if BP_indices[0,i]< h_width and h_width <= BP_indices[1,i] < (shape[1]-h_width):
            # Coordinates of edge bad pixels with central pixel at (0,0) 
            edge_bad_x=np.linspace(-h_width,h_width,width)
            edge_bad_y=np.linspace(-h_width,-1-BP_indices[0,i],h_width-BP_indices[0,i])
            edge_bad_x,edge_bad_y=np.meshgrid(edge_bad_x,edge_bad_y)
            edge_bad=np.vstack((np.reshape(edge_bad_y,edge_bad_y.size),np.reshape(edge_bad_x,edge_bad_x.size)))
            incomplete_kernel=GPR_Kernel(a,h,sig_data,width=width,close_badpix=np.append(edge_bad,close_bad,axis=1))
            # Fill in zero if outside of the actual image
            img=np.zeros((width,width))
            img[(h_width-BP_indices[0,i]):,:]=image[0:(BP_indices[0,i]+h_width+1), \
                                                          (BP_indices[1,i]-h_width):(BP_indices[1,i]+h_width+1)]
            # Convolution
            fixed_pix[i]=np.sum(incomplete_kernel*img)
        # Case 3: Bottom Edge
        if (shape[0]-h_width)<=BP_indices[0,i]<shape[0] and h_width <= BP_indices[1,i] < (shape[1]-h_width):
            # Coordinates of edge bad pixels with central pixel at (0,0) 
            edge_bad_x=np.linspace(-h_width,h_width,width)
            edge_bad_y=np.linspace(shape[0]-BP_indices[0,i],h_width,h_width-shape[0]+BP_indices[0,i]+1)
            edge_bad_x,edge_bad_y=np.meshgrid(edge_bad_x,edge_bad_y)
            edge_bad=np.vstack((np.reshape(edge_bad_y,edge_bad_y.size),np.reshape(edge_bad_x,edge_bad_x.size)))
            incomplete_kernel=GPR_Kernel(a,h,sig_data,width=width,close_badpix=np.append(edge_bad,close_bad,axis=1))
             # Fill in zero if outside of the actual image
            img=np.zeros((width,width))
            img[:(h_width+shape[0]-BP_indices[0,i]),:]=image[(BP_indices[0,i]-h_width):shape[0],\
                                                                   (BP_indices[1,i]-h_width):(BP_indices[1,i]+h_width+1)]
            # Convolution
            fixed_pix[i]=np.sum(incomplete_kernel*img)
        # Case 4: Left Edge
        if h_width <= BP_indices[0,i] < (shape[0]-h_width) and BP_indices[1,i]< h_width:
            # Coordinates of edge bad pixels with central pixel at (0,0) 
            edge_bad_x=np.linspace(-h_width,-1-BP_indices[1,i],h_width-BP_indices[1,i])
            edge_bad_y=np.linspace(-h_width,h_width,width)
            edge_bad_x,edge_bad_y=np.meshgrid(edge_bad_x,edge_bad_y)
            edge_bad=np.vstack((np.reshape(edge_bad_y,edge_bad_y.size),np.reshape(edge_bad_x,edge_bad_x.size)))
            incomplete_kernel=GPR_Kernel(a,h,sig_data,width=width,close_badpix=np.append(edge_bad,close_bad,axis=1))
             # Fill in zero if outside of the actual image
            img=np.zeros((width,width))
            img[:,(h_width-BP_indices[1,i]):]=image[(BP_indices[0,i]-h_width):(BP_indices[0,i]+h_width+1),\
                                                          0:(BP_indices[1,i]+h_width+1)]
            # Convolution
            fixed_pix[i]=np.sum(incomplete_kernel*img)
        # Case 5: Right Edge
        if h_width <= BP_indices[0,i] < (shape[0]-h_width) and (shape[1]-h_width)<=BP_indices[1,i]<shape[1]:
            # Coordinates of edge bad pixels with central pixel at (0,0) 
            edge_bad_x=np.linspace(shape[1]-BP_indices[1,i],h_width,h_width-shape[1]+BP_indices[1,i]+1)
            edge_bad_y=np.linspace(-h_width,h_width,width)
            edge_bad_x,edge_bad_y=np.meshgrid(edge_bad_x,edge_bad_y)
            edge_bad=np.vstack((np.reshape(edge_bad_y,edge_bad_y.size),np.reshape(edge_bad_x,edge_bad_x.size))) 
            incomplete_kernel=GPR_Kernel(a,h,sig_data,width=width,close_badpix=np.append(edge_bad,close_bad,axis=1))
            # Fill in zero if outside of the actual image
            img=np.zeros((width,width))
            img[:,:(h_width+shape[1]-BP_indices[1,i])]=image[(BP_indices[0,i]-h_width):(BP_indices[0,i]+h_width+1),\
                                                                   (BP_indices[1,i]-h_width):shape[1]]
            # Convolution
            fixed_pix[i]=np.sum(incomplete_kernel*img)
        # Case 6: On the Corners:
        if not (h_width <= BP_indices[0,i] < (shape[0]-h_width) or h_width <= BP_indices[1,i] < (shape[1]-h_width)):
            # Fill in a fixed value due to lack of information VS the complexity of matrix inversion
            fixed_pix[i]=fill
        residual[i]=image[BP_indices[0,i],BP_indices[1,i]]-fixed_pix[i]
        GPR_fixed_images[BP_indices[0,i],BP_indices[1,i]]=fixed_pix[i]
    return  GPR_fixed_images, residual
#%%
def GPR_training(image,TS,sig_data=1,K=Squared_Expo,width=9):
    # Supports images with bad pixels labeled as NaN
    if TS is "asnan":
        mask=np.isnan(image)
        # Convert Mask to indices of trainer pixels
        TS_indices=np.asarray(np.nonzero(mask))
        # Replace NaN with a finite value for the training process
        image[mask]=1
    # Supports trainer pixels indicated by a boolean mask
    elif TS.dtype == bool:
        TS_indices=np.asarray(np.nonzero(TS))
    else:
        TS_indices=TS
    shape=image.shape
    Ntrain=TS_indices.shape[1]
    img=np.zeros((TS_indices.shape[1],width**2))
    for i in range (Ntrain):
        # Avoid training on pixels on the edges
        if (width-1)/2 <= TS_indices[0,i] < (shape[0]-(width-1)/2) and (width-1)/2 <= TS_indices[1,i] < (shape[1]-(width-1)/2):
            # Construct the matrix of pixel values used in the convolution
            img_2D=image[(TS_indices[0,i]-int((width-1)/2)):(TS_indices[0,i]+int((width+1)/2)), \
                                          (TS_indices[1,i]-int((width-1)/2)):(TS_indices[1,i]+int((width+1)/2))]
            # Reshape into a row vector of the image matrix. The middle entry is the original pixel value.
            img[i]=np.reshape(img_2D,[width**2])
    # Function that computes the mean abs residual to minimize 
    def GPR_residual(para,full_residual=False):
        kernel=np.reshape(GPR_Kernel(para[0]**2,para[1]**2,sig_data,K=K,width=width),width**2)
        # Convolve all trainer pixels at the same time using matrix-vector multiplication
        convolved_pix=np.dot(img,kernel)
        residual=(img[:,int((width**2-1)/2)]-convolved_pix)
        if full_residual:
            return np.mean(np.abs(residual)),residual
        else:
            return np.mean(np.abs(residual))
    GPR_para=optimize.minimize(GPR_residual,[1,1],method="Powell").x
    return GPR_para**2,GPR_residual(GPR_para,full_residual=True)
#%%
def GPR_image_fix(image,BP,width=9,K=Squared_Expo):
    # Supports images with bad pixels labeled as NaN
    if BP is "asnan":
        mask=np.isnan(image)
        # Convert Mask to indices of bad pixels
        BadPix=np.asarray(np.nonzero(mask))
        # Replace NaN with a finite value for the training process
        image[mask]=1
    # Supports bad pixels indicated by a boolean mask
    elif BP.dtype == bool:
        BadPix=np.asarray(np.nonzero(BP))
    else:
        BadPix=BP
    BadPix=np.unique(BadPix,axis=1)
    # Estimate background distribution
    im_max=np.amax(image)
    bg_mean=np.median(image)
    bg_std=np.median(np.abs(image-np.median(image)))
    # Find out the brighter pixels
    BrightPix=np.logical_and(image>bg_mean+5*bg_std,image<im_max/5)
    # Use the brighter pixels as the training set
    para,residual=GPR_training(image,BrightPix,width=width,K=K)
    fixed_im,residual=GPR_fix(para[0],para[1],image,BadPix,width=width,K=K,fill=bg_mean)
    return fixed_im,para
