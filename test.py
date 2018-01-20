from os import listdir,mkdir
from shutil import copyfile


mkdir("train_data")
mkdir("test_data")

path = "/home/phuocluu/project_cat_content/python/data/data/thoisu/data"
folder = listdir(path)
for _ in range(len(folder)) :
    if _ % 4 != 0 :
        copyfile(path + "/" + folder[_], "train_data/" + folder[_])
    else :
    #if _ % 3 != 0 and _ % 4 == 0 :
        copyfile(path + "/" + folder[_], "test_data/" + folder[_])

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot([1,2,3])
# plt.show()
