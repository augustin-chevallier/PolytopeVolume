import numpy as np

d = 100

n = d*5

A = []
b = []

for i in range(n):
    vec = np.random.normal(0,1,size=(d))
    vec = vec/np.linalg.norm(vec)
    print(vec)
    A.append(vec)
    b.append(1)

for i in range(d):
    vec = np.zeros(d)
    vec[i] = 1
    A.append(np.copy(vec))
    b.append(1)
    vec[i] = -1
    A.append(np.copy(vec))
    b.append(1)

A = np.array(A)
#A = np.transpose(A)
print(A)
print(b)
np.savetxt(str(d)+"A.txt",A)
np.savetxt(str(d)+"b.txt",b)