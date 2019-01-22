import h5py as h5

files = ["output_1.h5", "output_2.h5", "output_3.h5", "output_4.h5", "output_5.h5", "output_6.h5", "output_7.h5", "output_8.h5", "output_9.h5", "output_10.h5"]
datasets = ["electrons", "muons", "tracks"]

data = {}
for key in datasets:
    data[key] = []

for file_name in files:
    file_data = h5.File("Outputs/" + file_name)
    for key in datasets:
        data[key] += list(file_data[key][()])

output = h5.File("combined_output.h5", "w")
for key in datasets:
    output.create_dataset(key, data=data[key])
