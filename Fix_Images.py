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
    # Check if separate h_x and h_y are used
    if type(h)==list:
        return a**2*np.exp(-(_x1-_x2)**2/(2*h[0]**2)+(_y1-_y2)**2/(2*h[1]**2))
    else:
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
def GPR_Kernel (a,h,sig_data=1,K=Squared_Expo,close_BP=None,width=9,badpix=[4,4]):
    if close_BP is None:
        close_BP=np.zeros((width,width),dtype=bool)
        close_BP[badpix[0],badpix[1]]=True
    good_pix=~close_BP
    # Change to coordinate where the bad pixel to be fixed is at (0,0)
    x=np.linspace(-badpix[1],close_BP.shape[1]-badpix[1]-1,close_BP.shape[1])
    y=np.linspace(-badpix[0],close_BP.shape[0]-badpix[0]-1,close_BP.shape[0])
    x,y=np.meshgrid(x,y)
    Cov_data=np.identity(x[good_pix].size)*sig_data**2    
    Kinv=np.linalg.inv(K(x[good_pix],x[good_pix],y[good_pix],y[good_pix],a,h)+Cov_data)
    Kernel=np.zeros(close_BP.shape)
    Kernel[good_pix]=np.dot(K(x[good_pix],0,y[good_pix],0,a,h),Kinv)[0]
    # Normalization
    Kernel/=np.sum(Kernel)
    # Reshape back to 2D
    Kernel=np.reshape(Kernel,close_BP.shape)
    return Kernel
#%%
def GPR_fix(a,h,image,BP,sig_data=1,K=Squared_Expo,width=9):
    # Make a copy of the image
    Image=image.copy()
    shape=Image.shape
    # Supports images with bad pixels labeled as NaN
    if BP is "asnan":
        BP_mask=np.isnan(Image)
        # Convert Mask to indices of bad pixels
        BP_indices=np.asarray(np.nonzero(BP_mask))
    # Supports bad pixels indicated by a boolean mask
    elif BP.dtype == bool:
        BP_mask=BP.copy()
        BP_indices=np.asarray(np.nonzero(BP))
    else:
        BP_mask=np.zeros(image.shape,dtype=bool)
        BP_mask[BP[0],BP[1]]=True
        BP_indices=BP.copy()
    Nbad=BP_indices.shape[1]
    fixed_pix=np.zeros(Nbad)
    GPR_fixed_image=image.copy()
    h_width=width//2
    # Precompute the kernel for the case of no additional bad pixels
    perfect_kernel=GPR_Kernel(a,h,sig_data=sig_data,K=K,width=width,badpix=[h_width,h_width])
    for i in range (Nbad):
        img=Image[max(0,BP_indices[0,i]-h_width):BP_indices[0,i]+h_width+1,\
                  max(0,BP_indices[1,i]-h_width):BP_indices[1,i]+h_width+1]
        submask=BP_mask[max(0,BP_indices[0,i]-h_width):BP_indices[0,i]+h_width+1,\
                        max(0,BP_indices[1,i]-h_width):BP_indices[1,i]+h_width+1]
        if np.sum(submask)==1 and submask.size==width**2:
            kernel=perfect_kernel
        else:    
            bp=np.array([BP_indices[0,i]-max(0,BP_indices[0,i]-h_width),BP_indices[1,i]-max(0,BP_indices[1,i]-h_width)])
            kernel=GPR_Kernel(a,h,sig_data=sig_data,K=K,close_BP=submask,badpix=bp)
        fixed_pix[i]=np.sum(kernel*img)
        GPR_fixed_image[BP_indices[0,i],BP_indices[1,i]]=fixed_pix[i]
    return  GPR_fixed_image
#%%
def GPR_training(image,TS,sig_data=1,K=Squared_Expo,width=9,init_guess=np.array([1,1])):
    # Make a copy of the image
    Image=image.copy()
    # Supports trainer pixels indicated by a boolean mask
    if TS.dtype == bool:
        TS_indices=np.asarray(np.nonzero(TS))
    else:
        TS_indices=TS
    shape=Image.shape
    Ntrain=TS_indices.shape[1]
    img=np.zeros((TS_indices.shape[1],width**2))
    h_width=width//2
    for i in range (Ntrain):
        # Avoid training on pixels on the edges
        if h_width <= TS_indices[0,i] < (shape[0]-h_width) and h_width <= TS_indices[1,i] < (shape[1]-h_width):
            # Construct the matrix of pixel values used in the convolution
            img_2D=Image[(TS_indices[0,i]-h_width):(TS_indices[0,i]+h_width+1), \
                                          (TS_indices[1,i]-h_width):(TS_indices[1,i]+h_width+1)]
            # Check if nan values are included
            if np.any(np.isnan(img_2D)):
                continue
            # Reshape into a row vector of the image matrix. The middle entry is the original pixel value.
            img[i]=np.reshape(img_2D,[width**2])
    # Function that computes the mean abs residual to minimize 
    def GPR_residual(para,full_residual=False):
        if init_guess.size==2:
            kernel=np.reshape(GPR_Kernel(para[0]**2,para[1]**2,sig_data=sig_data,K=K,width=width,badpix=[h_width,h_width]),width**2)
        else:
            kernel=np.reshape(GPR_Kernel(para[0]**2,[para[1]**2,para[2]**2],sig_data=sig_data,K=K,\
                                         width=width,badpix=[h_width,h_width]),width**2)
        # Convolve all trainer pixels at the same time using matrix-vector multiplication
        convolved_pix=img@kernel
        residual=(img[:,width**2//2]-convolved_pix)
        if full_residual:
            return np.mean(np.abs(residual)),residual
        else:
            return np.mean(np.abs(residual))
    para=optimize.minimize(GPR_residual,init_guess**(1/2),method="Powell").x
    return para**2,GPR_residual(para,full_residual=True)
#%%
def GPR_image_fix(image,BP,sig_clip=10,max_clip=5,sig_data=1,width=9,K=Squared_Expo,init_guess=np.array([1,1])):
    # Make a copy of the image
    Image=image.copy()
    # Remove repeating indices
    if not (BP is "asnan" or BP.dtype==bool):
        BP=np.unique(BP,axis=1)
    # Replace NaN values with 1 to naturally exclude them from the training set
    if BP is "asnan":
        Image[np.isnan(Image)]=1
    # Estimate background distribution
    im_max=np.amax(Image)
    bg_mean=np.median(Image)
    bg_std=np.median(np.abs(Image-np.median(Image)))
    # Find out the brighter pixels
    BrightPix=np.logical_and(Image>bg_mean+sig_clip*bg_std,Image<im_max/max_clip)
    # Use the brighter pixels as the training set
    para,residual=GPR_training(Image,BrightPix,sig_data=sig_data,width=width,K=K,init_guess=init_guess)
    if init_guess.size==2:
        fixed_im=GPR_fix(para[0],para[1],Image,BP,sig_data=sig_data,width=width,K=K)
    else:
        fixed_im=GPR_fix(para[0],[para[1],para[2]],Image,BP,sig_data=sig_data,width=width,K=K)
    return fixed_im,para
