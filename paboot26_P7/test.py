import numpy as np

test_arr = np.array(([1,2],[3,4],[5,6]))
new_arr = test_arr.mean(axis=0)
print(new_arr)
print(new_arr.shape)

