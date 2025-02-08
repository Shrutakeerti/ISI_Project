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

masker = NiftiMapsMasker(
    maps_img=atlas_filename,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    verbose=5,
)

time_series = masker.fit_transform(data.func[0], confounds=data.confounds)

from sklearn.covariance import GraphicalLassoCV

estimator = GraphicalLassoCV()
estimator.fit(time_series)

from nilearn import plotting

# Display the covariance

# The covariance can be found at estimator.covariance_
plotting.plot_matrix(
    estimator.covariance_,
    labels=labels,
    figure=(9, 7),
    vmax=1,
    vmin=-1,
    title="Covariance",
)

from nilearn import plotting

# Display the covariance

# The covariance can be found at estimator.covariance_
plotting.plot_matrix(
    estimator.covariance_,
    labels=labels,
    figure=(9, 7),
    vmax=1,
    vmin=-1,
    title="Covariance",
)

from nilearn import plotting

# Display the covariance

# The covariance can be found at estimator.covariance_
plotting.plot_matrix(
    estimator.covariance_,
    labels=labels,
    figure=(9, 7),
    vmax=1,
    vmin=-1,
    title="Covariance",
)

from nilearn import plotting

# Display the covariance


plotting.plot_matrix(
    -estimator.precision_,
    labels=labels,
    figure=(9, 7),
    vmax=1,
    vmin=-1,
    title="Sparse inverse covariance",
)

from sklearn.covariance import GraphicalLassoCV

estimator = GraphicalLassoCV()
estimator.fit(time_series)

from nilearn import plotting

# Display the covariance

# The covariance can be found at estimator.covariance_
plotting.plot_matrix(
    estimator.covariance_,
    labels=labels,
    figure=(9, 7),
    vmax=1,
    vmin=-1,
    title="Covariance",
)


