# L = cholesky(A)
# z = np.random.normal(0, 1, size=(3, 1))
# print(z)
# print("corr1")
# print(np.dot(L, z))
# z2 = np.random.normal(0, 1, size=(3, 1))
# z3 = np.append(z, z2, axis=1)
# print(np.dot(L, z3))
# print(np.dot(L, z3).transpose())
# a = np.array([1, 2, 3])
# print(np.tile(a, (4,1)))
import numpy as np
import pandas as pd

a = pd.DataFrame([])
b = pd.Series([50,50,50])
a = a.append(b,ignore_index=True)
a = a.append(b,ignore_index=True)
print(a)