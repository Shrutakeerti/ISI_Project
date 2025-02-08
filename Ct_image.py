import numpy as np 
import numpy.ma as ma
import matplotlib.pyplot as plt

# Both images are loaded from a dicom. Both are numpy arrays of (512,512) 
Image1 = readimage(path)
Image2 = readimage(path)
# Create image 2 mask
mask = ma.masked_where(Image2>0, Image2)
Image2_mask = ma.masked_array(Image2,mask)

# Plot images
plt.figure(dpi=300)
y, x = np.mgrid[1:513,1:513]
plt.axes().set_aspect('equal', 'datalim')
plt.set_cmap(plt.gray())
plt.pcolormesh(x, y, Image1,cmap='gray')
plt.pcolormesh(x, y, Image2_mask,cmap='jet')
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.show()



from nilearn.maskers import NiftiMasker
from nilearn.input_data import NiftiMasker
from nilearn import datasets

atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas["maps"]
# Loading atlas data stored in 'labels'
labels = atlas["labels"]

# Loading the functional datasets
data = datasets.fetch_development_fmri(n_subjects=1)

# print basic information on the dataset
print(f"First subject functional nifti images (4D) are at: {data.func[0]}")
from nilearn.maskers import NiftiMapsMasker

