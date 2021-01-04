# -*- coding: utf-8 -*-
"""
"""
#%%
import numpy as np
from scipy import optimize
from math import *
from warnings import warn
from astropy.utils.exceptions import AstropyUserWarning
#%%
def Squared_Expo(x1,x2,y1,y2,a,h):
    # Check if separate h_x and h_y are used
    if type(h)==list or type(h)==np.ndarray:
        if len(h) != 2:
            raise ValueError("For a stretched kernel, h must have exactly two values (h_x and h_y)")
        else:
            return a**2*np.exp(-(x1[:, np.newaxis] - x2[np.newaxis, :])**2/(2*h[0]**2)-\
                               (y1[:, np.newaxis] - y2[np.newaxis, :])**2/(2*h[1]**2))
    else:
        return a**2*np.exp(-((x1[:, np.newaxis] - x2[np.newaxis, :])**2+(y1[:, np.newaxis] - y2[np.newaxis, :])**2)*(1/(2*h**2)))
#%%
def Ornstein_U(x1,x2,y1,y2,a,h):
    if type(h)==list or type(h)==np.ndarray:
        if len(h) != 2:
            raise ValueError("For a stretched kernel, h must have exactly two values (h_x and h_y)")
        else:
            return a**2*np.exp(-np.abs(x1[:,np.newaxis]-x2[np.newaxis,:])/h[0]-np.abs(y1[:,np.newaxis]-y2[np.newaxis,:])/h[1])
    else:
        return a**2*np.exp(-np.sqrt((x1[:,np.newaxis]-x2[np.newaxis,:])**2+(y1[:,np.newaxis]-y2[np.newaxis,:])**2)/h)
#%%
def Periodic(x1,x2,y1,y2,a,h):
    if type(h)==list or type(h)==np.ndarray:
        if len(h) != 2:
            raise ValueError("For a stretched kernel, h must have exactly two values (h_x and h_y)")
        else:
            return a**2*np.exp(-2*(np.sin(x1[:,np.newaxis]-x2[np.newaxis,:])**2/h[0]**2\
                               +np.sin(y1[:,np.newaxis]-y2[np.newaxis,:])**2/h[1]**2))
    return a**2*np.exp(-2*np.sin(np.sqrt((x1[:,np.newaxis]-x2[np.newaxis,:])**2+(y1[:,np.newaxis]-y2[np.newaxis,:])**2)/2)**2/h**2)
#%%
def GPR_Kernel (a,h,sig_data=1,K=Squared_Expo,close_BP=None,width=9,badpix=None,x_grid=None,y_grid=None):
    if badpix is None:
        badpix=[width//2,width//2]
    if close_BP is None:
        close_BP=np.zeros((width,width),dtype=bool)
        close_BP[badpix[0],badpix[1]]=True
    if close_BP[badpix[0],badpix[1]]==False:
        raise ValueError("close_BP must be True at the location of badpix.")
    good_pix=~close_BP
    if x_grid is None and y_grid is None:
        x=np.linspace(0,close_BP.shape[1]-1,close_BP.shape[1])
        y=np.linspace(0,close_BP.shape[0]-1,close_BP.shape[0])
        x_grid,y_grid=np.meshgrid(x,y)
    elif (x_grid is None and y_grid is not None) or (x_grid is not None and y_grid is None) or x_grid.shape !=y_grid.shape:
            raise ValueError("x_grid and y_grid do not have the same shape, or only one of them is given. "\
                             "Cannot constuct a grid of coordinates.")
    if close_BP.shape != x_grid.shape:
        raise ValueError("close_BP should have the same shape as x and y grids.")
    # Change to coordinate where the bad pixel to be fixed is at (0,0)
    x_grid=x_grid-badpix[1]
    y_grid=y_grid-badpix[0]
    X, Y = x_grid[good_pix], y_grid[good_pix]
    Cov_data=np.identity(X.size)*sig_data**2    
    Kinv=np.linalg.inv(K(X,X,Y,Y,a,h)+Cov_data)
    Kernel=np.zeros(close_BP.shape)
    Kernel[good_pix]=np.dot(Kinv,K(X,np.zeros(1),Y,np.zeros(1),a,h))[:,0]
    # Normalization
    Kernel/=np.sum(Kernel)
    # Reshape back to 2D
    Kernel=np.reshape(Kernel,close_BP.shape)
    return Kernel
#%%
def Interpolate(a,h,image,BP,sig_data=1,K=Squared_Expo,width=9):
    # Make a copy of the image
    Image=image.copy()
    shape=Image.shape
    # Supports images with bad pixels labeled as NaN
    if BP is "asnan":
        BP_mask=np.isnan(Image)
        # Convert Mask to indices of bad pixels
        BP_indices=np.asarray(np.nonzero(BP_mask))
        # Set nan pixels to zero to prevent convolution from giving nan,
        # The nan pixels will be assigned zero weights anyway so any finite value can be used here
        Image[BP_mask]=0
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
    # Preconstruct the meshgrid
    x=np.linspace(0,width-1,width)
    y=np.linspace(0,width-1,width)
    x,y=np.meshgrid(x,y)
    for i in range (Nbad):
        img=Image[max(0,BP_indices[0,i]-h_width):BP_indices[0,i]+h_width+1,\
                  max(0,BP_indices[1,i]-h_width):BP_indices[1,i]+h_width+1]
        submask=BP_mask[max(0,BP_indices[0,i]-h_width):BP_indices[0,i]+h_width+1,\
                        max(0,BP_indices[1,i]-h_width):BP_indices[1,i]+h_width+1]
        if np.sum(submask)==1 and submask.size==width**2:
            kernel=perfect_kernel
        else: 
            x_grid=x[:submask.shape[0],:submask.shape[1]]
            y_grid=y[:submask.shape[0],:submask.shape[1]]
            bp=np.array([BP_indices[0,i]-max(0,BP_indices[0,i]-h_width),BP_indices[1,i]-max(0,BP_indices[1,i]-h_width)])
            kernel=GPR_Kernel(a,h,sig_data=sig_data,K=K,close_BP=submask,badpix=bp,x_grid=x_grid,y_grid=y_grid)
        fixed_pix[i]=np.sum(kernel*img)
        if np.any(np.isnan(fixed_pix)):
            warn("Nan values detected post convolution. Use a larger kernel width.",AstropyUserWarning)
        GPR_fixed_image[BP_indices[0,i],BP_indices[1,i]]=fixed_pix[i]
    return  GPR_fixed_image
#%%
def GPR_Train(image,TS=None,sig_data=1,K=Squared_Expo,width=9,sig_clip=10,max_clip=5,init_guess=[1,1],upper_bound=True,mu=3000,tau=200):
    # Make a copy of the image
    Image=image.copy()
    if TS is None:
        # The default training set
        # Estimate background distribution
        im_max=np.nanmax(Image)
        bg_mean=np.nanmedian(Image)
        bg_std=np.nanmedian(np.abs(Image-bg_mean))
        # Find out the training set pixels
        # Igore the warnings from comparing nan values
        with np.errstate(invalid='ignore'):
            TS=np.logical_and(Image>bg_mean+sig_clip*bg_std,Image<im_max/max_clip)
        TS_indices=np.asarray(np.nonzero(TS))
    # Supports training set pixels indicated by a boolean mask
    elif TS.dtype == bool:
        TS_indices=np.asarray(np.nonzero(TS))
    # Supports training set pixels indicated by a 2*N matrix containing the pixel coordinates
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
                                          (TS_indices[1,i]-h_width):(TS_indices[1,i]+h_width+1)].copy()
            # Reshape into a row vector of the image matrix. The middle entry is the original pixel value.
            img[i]=np.reshape(img_2D,[width**2])
    # Remove Bad Pixels that appear in the image matrix 
    img=img[np.isfinite(np.sum(img,axis=1))]
    if img.size==0:
        raise ValueError("The training set is empty or all training set pixels are close to bad pixels. " \
                         "Use smaller clipping parameters.")
    # Function that computes the mean abs residual to minimize 
    def Residual(para,full_residual=False):
        # Avoid singular matrices
        # h greater than a quarter pixel but less than width, a not below one, large a reasonable
        if para[0]**2<1 or np.any(para[1:]**2>width) or np.any(para[1:]**2<1/4):
            return 1e10
        if len(init_guess)==2:
            kernel=np.reshape(GPR_Kernel(para[0]**2,para[1]**2,sig_data=sig_data,K=K,width=width,badpix=[h_width,h_width]),width**2)
        else:
            kernel=np.reshape(GPR_Kernel(para[0]**2,[para[1]**2,para[2]**2],sig_data=sig_data,K=K,\
                                         width=width,badpix=[h_width,h_width]),width**2)
        # Convolve all training set pixels at the same time using matrix-vector multiplication
        convolved_pix=img@kernel
        residual=(img[:,width**2//2]-convolved_pix)
        if upper_bound:
            # Impose the soft upper bound on a to avoid large condition numbers while constructing kernels
            penalty=1+np.exp((para[0]**2-mu)/tau)
        else:
            # No upper bound on a
            penalty=1
        residual*=penalty
        if full_residual:
            return np.mean(np.abs(residual)),residual
        else:
            return np.mean(np.abs(residual))
    para=optimize.minimize(Residual,np.sqrt(init_guess),method="Powell").x
    return para**2,Residual(para,full_residual=True),TS
#%%
def Fix_Image(image,BP,TS=None,sig_clip=10,max_clip=5,sig_data=1,width=9,K=Squared_Expo,init_guess=[1,1],bad_to_nan=True):
    # Make a copy of the image
    Image=image.copy()
    # Remove repeating indices
    if not (BP is "asnan" or BP.dtype==bool):
        BP=np.unique(BP,axis=1)
    # Check if bad pixels are wished to be included in the training
    if bad_to_nan:
        current_BP="asnan"
        if not (BP is "asnan") and BP.dtype==bool:
            # Make the bad pixels nan to avoid them in the training set
            Image=Image.astype(float)
            Image[BP]=np.nan
        if not (BP is "asnan" or BP.dtype==bool):
            Image=Image.astype(float)
            Image[BP[0],BP[1]]=np.nan
    else:
        current_BP=BP.copy()
    para,residual,TS=GPR_Train(Image,TS=TS,sig_data=sig_data,width=width,K=K,sig_clip=sig_clip,max_clip=max_clip,init_guess=init_guess)
    if len(init_guess)==2:
        fixed_im=Interpolate(para[0],para[1],Image,current_BP,sig_data=sig_data,width=width,K=K)
    else:
        fixed_im=Interpolate(para[0],[para[1],para[2]],Image,current_BP,sig_data=sig_data,width=width,K=K)
    return fixed_im,para,TS
