# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:22:53 2021

@author: hyzha
"""

import astrofix
import numpy as np
import pytest
import os
import requests
from astropy.io import fits

def test_kernel_construction():
    for width in np.linspace(3,11,5):
        width=int(width)
        kernel=astrofix.GPR_Kernel(20,1,width=width)
        assert np.isclose(np.sum(kernel),1)
        assert kernel.shape==(width,width)
        assert kernel[width//2,width//2]==0
        
def test_training_results():
    file='astrofix/tests/cpt0m407-kb84-20200917-0147-e91.fits.fz'
    test_im= fits.open(file)[1].data
    corr_para=[3.3935965688743535,0.9324235469929149]
    para,resi,TS=astrofix.GPR_Train(test_im,max_clip=1)
    assert np.allclose(para,corr_para)
    assert np.isclose(np.count_nonzero(TS),281585)
        
def test_fixing_image():
    file='astrofix/tests/cpt0m407-kb84-20200917-0147-e91.fits.fz'
    test_im= fits.open(file)[1].data
    BP_mask=np.loadtxt("astrofix/tests/Sample_Bad_Pixel_Mask.gz")
    BP_mask=BP_mask.astype(bool)
    corr_resi=np.loadtxt("astrofix/tests/Sample_Residual.gz")
    img=test_im.copy()
    img[BP_mask]=np.nan
    fixed_img,para,TS=astrofix.Fix_Image(img,"asnan",max_clip=1)
    assert np.allclose(fixed_img-test_im,corr_resi)
    assert np.isclose(np.count_nonzero(TS),278951)
