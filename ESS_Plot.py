import numpy as np 
import matplotlib.pyplot as plt


def get_data(polytope,sampler,dim):

    filename = "_" + sampler + '_' + polytope + "_" + str(dim) + ".txt"
    print(filename)

    with open('ESS' + filename, 'r') as f:
        l = [[float(num) for num in line.strip().split(' ')] for line in f]
    l = np.array(l)
    a_vals = l[:,0]
    ess = l[:,1:]

    with open('Times' + filename, 'r') as f:
        l = [[float(num) for num in line.strip().split(' ')] for line in f]
    times = np.array(l)

    with open('NumSamples' + filename, 'r') as f:
        l = [[float(num) for num in line.strip().split(' ')] for line in f]
    nsamples = np.array(l)

    with open('NumOracles' + filename, 'r') as f:
        l = [[float(num) for num in line.strip().split(' ')] for line in f]
    noracles = np.array(l)

    #print(l)
    #print(l[:,0])
    #print(ess)
    return (a_vals,ess,times,nsamples,noracles)


print(get_data("cube","BPSO",50))

data = get_data("cube","BPSO",50)

ess_per_sec = data[1]/data[2]
print(ess_per_sec)