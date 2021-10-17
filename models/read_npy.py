

import numpy as np

a = np.load('save_2021-09-09.npy')
# a = np.transpose(a, (0,3, 1, 2))

print(a.shape)

# a = np.asarray(a)
# np.savetxt("foo.csv", a, delimiter=",")


# for near_metric in range(a.shape[0]):

#     array = np.transpose(a[near_metric], (2, 0, 1))

#     for i in range(2):
        
#         array1 = np.asarray(array[i])
#         np.savetxt("foo_"+str(near_metric)+"_"+str(i)+".csv", array1, delimiter=",")


from xml.dom import minidom

doc = minidom.parse('model.svg')  # parseString also exists
path_strings = [path.getAttribute('d') for path
                in doc.getElementsByTagName('path')]
doc.unlink()

print(path_strings)
print(doc)
