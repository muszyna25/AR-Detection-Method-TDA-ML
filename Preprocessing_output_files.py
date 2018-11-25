import os
import numpy as np
import datetime
import sys

# General settings.
path_to_artmip_data = "/global/cscratch1/sd/muszyng/MERRA2_2D_data_3_hourly_results/MERRA2_2D_data_3_hourly_shifted"

# Go to data dir. 
os.chdir(path_to_artmip_data)
print(os.getcwd())

# Create an empty data matrix.
dataset = np.array([])

# Get list of directories with data for each year, e.g. 2001, ...., 2017.
dirList = os.listdir("./")
dirList = sorted(dirList, key=lambda x: datetime.datetime.strptime(x, '%Y'))
print(dirList)

# Go through all directories, read and copy data to data matrix.

nb_dimensions = 60 #number of features == max IWV value.

for d in dirList:
    # Check if directory.
    if os.path.isdir(d) == True:
        print("Year: ", d)

        os.chdir(d + "/volume")

        # Get list of files in the current directory.
        curr_dir = os.getcwd()
        print(curr_dir)

        content_list = []
        for content in os.listdir('.'):
            if content.endswith(".txt"):
                content_list.append(content)

        # Read content of the data files and merge it into one matrix.
        names =""
        nb_samples = len(content_list)

        matrix = np.zeros((nb_samples, nb_dimensions), dtype=int)

        # Read file.
        for l in range(0,len(content_list)):
            filename = content_list[l]
            names = names + filename[14:-4] + "_"
            x, y, z, i0, i1 = np.loadtxt(filename, delimiter=',', unpack=True)

            n = len(x)
            tmp = 0
            all_volumes = []
            for i in range(0,n,1):
                if(tmp != x[i]):
                    tmp = x[i]
                    all_volumes.append(y[i])
                else:
                    continue

                for j in range(i,n,1):
                    if(tmp == x[j]):
                        all_volumes.append(y[j])

                value = max(all_volumes)
                matrix[l, int(tmp)] = value

        # Store array one above each other.
        print ('Generated dataset (matrix): \n')
        print (matrix)
        if(dataset.size):
            dataset = np.vstack([dataset, matrix]) # When intilizing the matrix.
        else:
            dataset = matrix
        print ('Dataset: \n')
        print(dataset)
        os.chdir("../../")

print(dataset.shape)

# Save data to one large text file.
np.savetxt("MERRA2_data.txt", dataset, fmt="%d")
