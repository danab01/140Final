#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import hcipy
import random


# In[2]:


# Parameters for the pupil function
pupil_diameter = 0.019725 # m
gap_size = 90e-6 # m
num_rings = 3
segment_flat_to_flat = (pupil_diameter - (2 * num_rings + 1) * gap_size) / (2 * num_rings + 1)
focal_length = 1 # m

# Parameters for the simulation
num_pix = 512
wavelength = 638e-9
num_airy = 20
sampling = 4
norm = False


# In[3]:


# HCIPy grids and propagator
pupil_grid = hcipy.make_pupil_grid(dims=num_pix, diameter=pupil_diameter)

focal_grid = hcipy.make_focal_grid(sampling, num_airy,
                                   pupil_diameter=pupil_diameter,
                                   reference_wavelength=wavelength,
                                   focal_length=focal_length)
focal_grid = focal_grid.shifted(focal_grid.delta / 2)

prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)


# In[4]:


# Define function from rad of phase to m OPD
def aber_to_opd(aber_rad, wavelength):
    aber_m = aber_rad * wavelength / (2 * np.pi)
    return aber_m

aber_rad = 4.0

print('Aberration: {} rad'.format(aber_rad))
print('Aberration: {} m'.format(aber_to_opd(aber_rad, wavelength)))


# In[107]:


focal_number = 2*num_airy*sampling

def make_FPM(focal_grid, N=focal_number):
    
    array = np.array([[0.0]*N]*N)
    
    for i in range(0,N):
        for j in range (0, N):
            x = abs(int(N/2) - i)
            y = abs(int(N/2) - j)
            r = np.sqrt(x**2 + y**2)
            if r < 3.5*N/16 and r > 2*N/16:
                array[i][j] = 1.0
    
    array_flat = array.flatten()
    return hcipy.Field(array_flat, focal_grid)

annulus = make_FPM(focal_grid)
hcipy.imshow_field(annulus)
plt.show()


# In[100]:


def score_matrix3(aberrations, show=False):
    
    # Make segmented mirror
    aper, segments = hcipy.make_hexagonal_segmented_aperture(num_rings,
                                                         segment_flat_to_flat,
                                                         gap_size,
                                                         starting_ring=1,
                                                         return_segments=True)

    aper = hcipy.evaluate_supersampled(aper, pupil_grid, 1)
    segments = hcipy.evaluate_supersampled(segments, pupil_grid, 1)
    
    # Instantiate the segmented mirror
    hsm = hcipy.SegmentedDeformableMirror(segments)

    # Make a pupil plane wavefront from aperture
    wf = hcipy.Wavefront(aper, wavelength)
    
    hsm.flatten()
    
    # Get poke segments
    for i in range(0,len(aberrations)):
        if float(aberrations[i]) != 0:
            hsm.set_segment_actuators(i, aber_to_opd(aber_rad, wavelength) / (1/float(aberrations[i])), 0, 0)
    
    if show:
        plt.figure(figsize=(8,8))
        plt.title('OPD for HCIPy SM')
        hcipy.imshow_field(hsm.surface * 2, mask=aper, cmap='RdBu_r', vmin=-5e-7, vmax=5e-7)
        plt.colorbar()
        plt.show()
    
    ### PROPAGATE AND SCORE ###
    wf_fp_pistoned = hsm(wf)

    # Propagate from SM to image plane
    im_pistoned_hc = prop(wf_fp_pistoned)
    norm_hc = np.max(im_pistoned_hc.intensity)
    
    if show:
        # Display intensity in image plane

        hcipy.imshow_field(np.log10(im_pistoned_hc.intensity / norm_hc), cmap='inferno', vmin=-9)
        plt.title('Image plane after SM')
        plt.colorbar()
        plt.show()
        
        hcipy.imshow_field(np.log10((im_pistoned_hc.intensity / norm_hc)*(annulus)), cmap='inferno', vmin=-9)
        plt.title('Annular image plane region')
        plt.colorbar()
        plt.show()
        
    interested_field = (im_pistoned_hc.intensity / norm_hc)*(1-annulus)
    score = float(hcipy.Field.sum(interested_field)/hcipy.Field.sum(im_pistoned_hc.intensity / norm_hc))*100
    
    return score


# In[13]:


for x in range(6):
    aberrations = [random.gauss(0,0.2) for _ in range(36)]
    score = score_matrix3(aberrations, show=True)
    print(score)


# In[24]:


# Generate data

lines = []

for x in range(5000):
    aberrations = [random.gauss(0,0.25) for _ in range(36)]
    mask = [random.randint(0,1) for _ in range(36)]
    inputs = [a * b for a, b in zip(aberrations, mask)]
    score = score_matrix3(inputs)
    aberrations.append(score)
    
    lines.append(aberrations)


# In[25]:


import csv

with open('DM_data2.csv', mode='w') as DM_data2:
    writer = csv.writer(DM_data2, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    for l in lines:
        writer.writerow(l)


# In[30]:


clear = [0]*36
score_matrix3(clear)


# In[101]:


score = score_matrix3([-0.00850241, -0.0081833,  -0.00736596,  0.00134285, -0.00223289,  0.00450384,
  -0.00827297, -0.00959549,  0.01420154,  0.002233,    0.00046704,  0.00010382,
   0.00592388,  0.0122226,   0.00649502,  0.00089059,  0.00260084,  0.00740923,
  -0.00664559, -0.00330013,  0.00226148, -0.00504211, -0.00955859, -0.00385683,
  -0.00500073,  0.00067157, -0.00118563,  0.00732557,  0.00380876,  0.01321707,
   0.00511285,  0.00057616, -0.0011992,  -0.01020722, -0.00059352,  0.0017242], show=True)

print(score)


# In[106]:


score = score_matrix3([-0.03803418,  0.39940897,  0.01336813,  0.44190282,  0.42580876,
         0.29408893, -0.2301098 , -0.019265  , -0.05201268, -0.4703673 ,
         0.079392  ,  0.02565664,  0.40929827,  0.15467829,  0.18510732,
         0.05081817,  0.35514715,  0.5770618 , -1.0810735 ,  0.5966555 ,
         0.5633484 ,  0.42992193,  0.2670331 ,  0.5131791 ,  0.50659674,
         0.02949879,  0.05652304,  0.03926276,  0.4099731 , -0.34369206,
        -0.13623442,  0.441069  , -0.20265165,  0.34984908,  0.34766257,
         1.56564   ], show=True)
print(score)


# In[ ]:





# In[ ]:




