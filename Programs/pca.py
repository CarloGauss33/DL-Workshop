import numpy as np
from sklearn import decomposition 

# Define individual features
x1 = np.random.normal(size=250)
x2 = np.random.normal(size=250)
x3 = 2*x1 + 3*x2
x4 = 4*x1 - x2
x5 = x3 + 2*x4

X = np.c_[x1, x3, x2, x5, x4]   
pca = decomposition.PCA()
pca.fit(X)

variances = pca.explained_variance_
print('\nVariances in decreasing order:\n', variances)

thresh_variance = 0.8
num_useful_dims = len(np.where(variances > thresh_variance)[0])
print('\nNumber of useful dimensions:', num_useful_dims)

pca.n_components = num_useful_dims

X_new = pca.fit_transform(X)
print('\nShape before:', X.shape)
print('Shape after:', X_new.shape)