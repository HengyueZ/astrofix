# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from scipy import optimize
from warnings import warn
from astropy.utils.exceptions import AstropyUserWarning

def Squared_Expo(x1,x2,y1,y2,a,h):
    """
    Squared_Expo(x1,x2,y1,y2,a,h)
    Compute the covariance matrix between points in set 1 and set 2, using the Squared Exponential covariance function. 
    
    x1: numpy array, shape (m,)
       An array of x-coodinates for points in set 1.
       
    x2: numpy array, shape (n,)
       An array of x-coodinates for points in set 2.
       
    y1: float numpy array, shape (m,)
       An array of y-coodinates for points in set 1.
       
    y2: float numpy array, shape (n,)
       An array of y-coodinates for points in set 2.
       
    a: float
       A parameter scaling the correlation between points.  
    
    h: float or shape (2,) array-like
       A parameter scaling the characteristic length over which the corrlation values vary. If h is a list or numpy array, 
       different characteristic scales will be used in the x and y directions, with h[0]=h_x and h[1]=h_y.
    
    Returns
    -------
    Cov: float numpy array
       The covariance matrix.
    """
    
    # Check if separate h_x and h_y are used
    if type(h)==list or type(h)==np.ndarray:
        if len(h) != 2:
            raise ValueError("For a stretched kernel, h must have exactly two values (h_x and h_y)")
        else:
            Cov=a**2*np.exp(-(x1[:, np.newaxis] - x2[np.newaxis, :])**2/(2*h[0]**2)-\
                               (y1[:, np.newaxis] - y2[np.newaxis, :])**2/(2*h[1]**2))
            return Cov
    else:
        Cov=a**2*np.exp(-((x1[:, np.newaxis] - x2[np.newaxis, :])**2+(y1[:, np.newaxis] - y2[np.newaxis, :])**2)*(1/(2*h**2)))
        return Cov

def Ornstein_U(x1,x2,y1,y2,a,h):
    """
    Ornstein_U(x1,x2,y1,y2,a,h)
    Compute the covariance matrix between points in set 1 and set 2, using the Ornstein-Uhlenbeck covariance function. 
    
    Parameters
    ----------
    x1: numpy array, shape (m,)
       An array of x-coodinates for points in set 1.
       
    x2: numpy array, shape (n,)
       An array of x-coodinates for points in set 2.
       
    y1: float numpy array, shape (m,)
       An array of y-coodinates for points in set 1.
       
    y2: float numpy array, shape (n,)
       An array of y-coodinates for points in set 2.
       
    a: float
       A parameter scaling the correlation between points.  
    
    h: float or shape (2,) array-like
       A parameter scaling the characteristic length over which the corrlation values vary. If h is a list or numpy array, 
       different characteristic scales will be used in the x and y directions, with h[0]=h_x and h[1]=h_y.
    
    Returns
    -------
    Cov: float numpy array
       The covariance matrix.
    """
    
    if type(h)==list or type(h)==np.ndarray:
        if len(h) != 2:
            raise ValueError("For a stretched kernel, h must have exactly two values (h_x and h_y)")
        else:
            Cov=a**2*np.exp(-np.abs(x1[:,np.newaxis]-x2[np.newaxis,:])/h[0]-np.abs(y1[:,np.newaxis]-y2[np.newaxis,:])/h[1])
            return Cov
    else:
        Cov=a**2*np.exp(-np.sqrt((x1[:,np.newaxis]-x2[np.newaxis,:])**2+(y1[:,np.newaxis]-y2[np.newaxis,:])**2)/h)
        return Cov

def Periodic(x1,x2,y1,y2,a,h):
    """
    Periodic(x1,x2,y1,y2,a,h)
    Compute the covariance matrix between points in set 1 and set 2, using the periodic covariance function. 
    
    Parameters
    ----------
    x1: numpy array, shape (m,)
       An array of x-coodinates for points in set 1.
       
    x2: numpy array, shape (n,)
       An array of x-coodinates for points in set 2.
       
    y1: float numpy array, shape (m,)
       An array of y-coodinates for points in set 1.
       
    y2: float numpy array, shape (n,)
       An array of y-coodinates for points in set 2.
       
    a: float
       A parameter scaling the correlation between points.  
    
    h: float or shape (2,) array-like
       A parameter scaling the characteristic length over which the corrlation values vary. If h is a list or numpy array, 
       different characteristic scales will be used in the x and y directions, with h[0]=h_x and h[1]=h_y.
    
    Returns
    -------
    Cov: float numpy array
       The covariance matrix.
    """
    
    if type(h)==list or type(h)==np.ndarray:
        if len(h) != 2:
            raise ValueError("For a stretched kernel, h must have exactly two values (h_x and h_y)")
        else:
            Cov=a**2*np.exp(-2*(np.sin(x1[:,np.newaxis]-x2[np.newaxis,:])**2/h[0]**2\
                                +np.sin(y1[:,np.newaxis]-y2[np.newaxis,:])**2/h[1]**2))
            return Cov
    else:
        Cov=a**2*np.exp(-2*np.sin(np.sqrt((x1[:,np.newaxis]-x2[np.newaxis,:])**2+(y1[:,np.newaxis]-y2[np.newaxis,:])**2)/2)**2/h**2)
        return Cov

def GPR_Kernel (a,h,sig_data=1,K=Squared_Expo,close_BP=None,width=9,badpix=None,x_grid=None,y_grid=None):
    """
    GPR_Kernel (a,h,sig_data=1,K=Squared_Expo,close_BP=None,width=9,badpix=None,x_grid=None,y_grid=None)
    Construct a GPR interpolation kernel.
    
    Parameters
    ----------  
    a: float
       Parameter in the covariance function; the correlation between points.  
    
    h: float or shape (2,) array-like
       Parameter in the covariance function; the characteristic scale over which the correlation values vary. 
       If h is an numpy array, different characteristic scales will be used in the x and y 
       directions, with h[0]=h_x and h[1]=h_y.  
       
    sig_data: float, optional
       Measurement noise, assumed to be uniform. The kernel depends only on the ratio 
       a/sig_data. Default: 1.  
       
    K: callable, optional
       The covariance function to be used. The built-in options are: Squared_Expo, Ornstein_U, and Periodic. 
       See the corresponding functions above for their definitions. Default: Squared_Expo.
       
    close_BP: boolean numpy array, optional
       A boolean mask that specifies the bad pixel locations in the kernel. Masked pixels will be 
       assigned zero weights. The shape of the kernel will be the same as the shape of the mask. 
       If no mask is provided, the kernel will be width*width with the only bad pixel located at 
       the center. Default: None.  
       
    width: int, optional
       The width of the default mask if close_BP is None. Default: 9.  
       
    badpix: shape (2,) int numpy array, optional
       Index of the bad pixel to be fixed in the close_BP mask. If not provided, it will be 
       [width//2,width//2]. Default: None.  
       
    x_grid: int numpy array, optional
       A 2d-array specifying the x-coordinates of the pixels. If not provided, it will be x going 
       from 0 to close_BP.shape[1]-1. Default: None.
       
    y_grid: int numpy array, optional
       A 2d-array specifying the y-coordinates of the pixels. Must be of the same shape as x_grid. 
       If not provided, it will be y going from 0 to close_BP.shape[0]-1. Default: None.
       
    Returns
    -------
    Kernel: float numpy array
       The constructed kernel.
    """
    
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

def Interpolate(a,h,image,BP,sig_data=1,K=Squared_Expo,width=9):
    """
    Interpolate(a,h,image,BP,sig_data=1,K=Squared_Expo,width=9)
    Correcting the values of bad pixels by interpolation.
    
    Parameters
    ----------
    a: float
       Same as in GPR_Kernel.
       
    h: float or shape (2,) array-like
       Same as in GPR_Kernel.
       
    image: numpy array 
       The image to be fixed.
       
    BP: {boolean numpy array, "asnan", shape (2,N_{bad}) int numpy array}
       The bad pixels to be fixed. Supports three different formats:  
       (1) A boolean mask with bad pixels having values of True. Must be of the same shape as image.    
       (2) The string "asnan", meaning that the bad pixels are labeled in the image by np.nan.   
       (3) A shape (2,N_{bad}) array, with the first row giving the y indices and the second row 
           giving the corresponding x indices of the bad pixels.
           
    sig_data: float, optional
       Same as in GPR_Kernel. Default: 1.
       
    K: callable, optional
       Same as in GPR_Kernel. Default: Squared_Expo.
    
    width: int, optional 
       Same as in GPR_Kernel. Default: 9.

    Returns
    -------
    GPR_fixed_image: float numpy array
       The fixed image.
    """
    
    # Make a copy of the image
    Image=image.copy()
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

def GPR_Train(image,TS=None,BP="asnan",sig_data=1,K=Squared_Expo,width=9,\
              sig_clip=10,max_clip=5,init_guess=[1,1],upper_bound=True,mu=3000,tau=200):
    """
    GPR_Train(image,TS=None,BP="asnan",sig_data=1,K=Squared_Expo,width=9,\
              sig_clip=10,max_clip=5,init_guess=[1,1],upper_bound=True,mu=3000,tau=200)
    Find the optimal kernel parameter values for a given image by minimizing the mean absolute residual of convolution.
    
    Parameters
    ----------
    image: numpy array
       The image to be trained on.  
       
    TS: boolean numpy arrary or shape (2,N_{train}) int numpy array, optional 
       The training set to be used for the optimization. Supports two different formats:  
       (1) A boolean mask with trainer pixels having values of True. Must be of the same shape as image.  
       (2) A shape (2, N_{train}) array, with the first row giving the y indices and the second row giving the 
           corresponding x indices of the trainer pixels.  
       If not provided, will construct the training set using sig_clip and max_clip.  
       Default: None.
       
    BP: {boolean numpy array, "asnan", shape (2,N_{bad}) int numpy array}, optional
       The bad pixels. If a training set pixel is too close to at least one of the bad pixels, it will be removed
       from the training set. Supports three different formats:  
       (1) A boolean mask with bad pixels having values of True. Must be of the same shape as image.   
       (2) The string "asnan", meaning that the bad pixels are labeled in the image by np.nan   
       (3) A shape (2, N_{bad}) array, with the first row giving the y indices and the second row giving the 
           corresponding x indices of the bad pixels.  
       Default: "asnan".
       
    sig_data: float, optional
       Same as in GPR_Kernel. Default: 1.  
    
    K: callable, optional
       Same as in GPR_Kernel. Default: Squared_Expo.  
    
    width: int, optional
       Same as in GPR_Kernel. Default: 9. 
    
    sig_clip: float, optional
       Pixels that are smaller than median + sig_clip * median absolute deviation of the image will not be 
       used in the training process.  Default: 10.  
       
    max_clip: float, optional
       Pixels that are greater than max(image)/max_clip will not be used in the training process. Default: 5.
        
    init_guess: array-like, optional
       Initial guess for the training process. By default, the 0th element gives the initial guess of a and the 
       1st element gives the initial guess of h. If the size of init_guess is 3, the training optimizes h_x and
       h_y separately instead of using h for all directions. In that case, the 1st element gives the initial 
       guess of h_x, and the 2nd element gives the initial guess of h_y. Default: [1,1].
        
    upper_bound: boolean, optional
       If True, will set a soft upper bound on a by multiplying the mean absolute residual with the penalty 
       function 1+exp[(a-mu)/tau]. Default: True.
      
    mu: float, optional
       A parameter in the penalty function scaling the soft upper bound on the value of a. Default: 3000.
       
    tau: float, optional
       A parameter in the penalty function affecting how much a can penetrate the soft upper bound. Default: 200.
       
    Returns
    -------
    para: array-like
       The optimized parameters, where a=para[0], and h=para[1].  
    
    Residual: list
       The residual of the optimization. The 0th element is the minimized mean absolute residual, and the 1st 
       element is the full residual vector. 
       
    TS: boolean numpy array
       The chosen training set.
    """
    
    # Make a copy of the image
    Image=image.copy()
    # Check if bad pixels are wished to be included in the training. If not, flag them as NaN
    if BP is not "asnan":
        Image=Image.astype(float)
        if BP.dtype==bool:
            Image[BP]=np.nan
        else:
            Image[BP[0],BP[1]]=np.nan
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

def Fix_Image(image,BP,TS=None,sig_clip=10,max_clip=5,sig_data=1,width=9,K=Squared_Expo,init_guess=[1,1],bad_to_nan=True):
    """
    Fix_Image(image,BP,TS=None,sig_clip=10,max_clip=5,sig_data=1,width=9,K=Squared_Expo,init_guess=[1,1],bad_to_nan=True):
    Fixing bad pixels on an image.
    
    Parameters
    ----------
    image: numpy array
       The image to be fixed.  
       
    BP: {boolean numpy array, "asnan", shape (2,N_{bad}) int numpy array}
       The bad pixels to be fixed. Supports three different formats: 
       (1) A boolean mask with bad pixels having values of True. Must be of the same shape as image.   
       (2) The string "asnan", meaning that the bad pixels are labeled in the image by np.nan   
       (3) A shape (2, N_{bad}) array, with the first row giving the y indices and the second row giving the corresponding 
           x indices of the bad pixels.      
       
    TS: boolean numpy arrary or shape (2,N_{train}) int numpy array, optional 
       The training set to be used for the optimization. Supports two different formats:  
       (1) A boolean mask with trainer pixels having values of True. Must be of the same shape as image.  
       (2) A shape (2, N_{train}) array, with the first row giving the y indices and the second row giving the 
           corresponding x indices of the trainer pixels.  
       If not provided, will construct the training set using sig_clip and max_clip.  
       Default: None.
       
    sig_clip: float, optional
       Pixels that are smaller than median + sig_clip * median absolute deviation of the image will not be used in 
       the training process.  Default: 10.  
       
    max_clip: float, optional
       Pixels that are greater than max(image)/max_clip will not be used in the training process. Default: 5.  
       
    sig_data: float, optional 
       Same as in GPR_Kernel. Default: 1.  
       
    width:  int, optional
       Same as in GPR_Kernel. Default: 9.  
       
    K: callable, optional
       Same as in GPR_Kernel. Default: Squared_Expo.  
    
    init_guess: array-like, optional
       Initial guess for the training process. By default, the 0th element gives the initial guess of a and the 1st 
       element gives the initial guess of h. If the size of init_guess is 3, the training optimizes h_x and h_y separately 
       instead of using h for all directions. In that case, the 1st element gives the initial guess of h_x, 
       and the 2nd element gives the initial guess of h_y. Default: [1,1].
       
    bad_to_nan: boolean, optional
       If True, bad pixels will have their values replaced by np.nan so that they will be excluded in the training process. Default: True
       
    Returns:
    -------
    fixed_im:  float numpy array
       The fixed image.  
    
    para: array-like
       The optimized parameters, where a=para[0], and h=para[1]. 
       
    TS: boolean numpy array
       The chosen training set.
    """
    
    # Make a copy of the image
    Image=image.copy()
    # Remove repeating indices
    if not (BP is "asnan" or BP.dtype==bool):
        BP=np.unique(BP,axis=1)
    # Check if bad pixels are wished to be included in the training. If not, flag them as NaN
    if bad_to_nan:
        Image=Image.astype(float)
        current_BP="asnan"
        if not (BP is "asnan") and BP.dtype==bool:
            # Make the bad pixels nan to avoid them in the training set
            Image[BP]=np.nan
        if not (BP is "asnan" or BP.dtype==bool):
            Image[BP[0],BP[1]]=np.nan
    else:
        current_BP=BP.copy()
    para,residual,TS=GPR_Train(Image,TS=TS,sig_data=sig_data,width=width,K=K,sig_clip=sig_clip,max_clip=max_clip,init_guess=init_guess)
    if len(init_guess)==2:
        fixed_im=Interpolate(para[0],para[1],Image,current_BP,sig_data=sig_data,width=width,K=K)
    else:
        fixed_im=Interpolate(para[0],[para[1],para[2]],Image,current_BP,sig_data=sig_data,width=width,K=K)
    return fixed_im,para,TS
