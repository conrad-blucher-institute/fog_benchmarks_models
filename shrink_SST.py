# Script to scale SST data from
#  (376 x 376) to (32 x 32)

import numpy as np
from skimage.transform import resize

dataDir   = "/data1/fog/fognn/Dataset/24HOURS/2D/"
initYear = 2009
lastYear = 2020
allYears = range(initYear, lastYear + 1)
pathPatternSSTin  = "NETCDF_SST_CUBE_{}_24.npz"
pathPatternSSTout = "NETCDF_SST_CUBE_{}_24_scaled.npz"

cubeFilesIn = [dataDir + "/" + pathPatternSSTin.format(y) for y in allYears]
cubeFilesOut = [dataDir + "/" + pathPatternSSTout.format(y) for y in allYears]

print("")
ncubes = len(cubeFilesIn)
for i in range(ncubes):
	print("Scaling cube {} / {}".format(i, ncubes))
	# Open cube
	cube = np.load(cubeFilesIn[i])
	cube = cube["arr_0"]
	print("Cube file in", cubeFilesIn[i])
	print("Cube shape in", cube.shape)

	slices = [np.zeros((32, 32, 1)) for j in range(cube.shape[0])]
	for c in range(cube.shape[0]):
		img = cube[c][:,:,0]
		# Resize cube
		scaled = resize(img, (32, 32))
		slices[c][:, :, 0] = scaled
	combo = np.array(slices)
	
	# Save cube
	np.savez(cubeFilesOut[i], combo)
	print("Cube file out", cubeFilesOut[i])
	print("Cube shape out", combo.shape)
	print("")
