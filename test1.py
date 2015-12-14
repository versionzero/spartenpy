
import sys
import numpy as np
import spartenpy as sp

L = 3
M = 4
N = 5

__test1 = np.array([[(0,  1, 0,  2),
                     [3,  0, 4,  0],
                     [0,  5, 6,  0]],
                    [[ 7, 0, 8,  0],
                     [ 0, 9, 10, 0],
                     [11, 0, 0,  12]]])

a = sp.array.frontal(__test1)
sta = sp.tensor.fromarray(a)
st = sp.coo_tensor(sta)

#st = sp.rand(L, M, N)
#st = sp.testten()

#print st.tensor.subs

#print st.tensor.subs[0]
#print st.tensor.subs[1]
#print st.tensor.subs[2]
#print st.tensor.vals

#print st

crsst = st.tocrs()

print "\nCRS"
print crsst.RO
print crsst.CO
print crsst.KO
print crsst.vals

ccsst = st.toccs()

print "\nCCS"
print ccsst.RO
print ccsst.CO
print ccsst.KO
print ccsst.vals

crsst = crsst.toccs()

print "\nCCS (from CRS)"
print crsst.RO
print crsst.CO
print crsst.KO
print crsst.vals

ctsst = st.tocts()

print "\nCTS"
print ctsst.RO
print ctsst.CO
print ctsst.KO
print ctsst.vals

# ECRS requires the input to be oriented to lateral slices.
# TODO: Re-order in the coo tensor to make this unessesary
# print __test1
a = sp.array.lateral(__test1)
#print a

sta = sp.tensor.fromarray(a)
st = sp.coo_tensor(sta)

ecrsst = st.toecrs()

print "\nECRS"
print ecrsst.R
print ecrsst.CK
print ecrsst.vals

print __test1
a = sp.array.vertical(__test1)
#a == __test1
print a

sta = sp.tensor.fromarray(a)
st = sp.coo_tensor(sta)

eccsst = st.toeccs()

print "\nECCS"
print eccsst.R
print eccsst.CK
print eccsst.vals

sys.exit()

#m = 3
#n = 4
#d = m * n
#i2 = arange(d).reshape(m, n).T.flatten()
#Tmn = zeros((d, d))
#Tmn[arange(d), i2] = 1
#print Tmn




t = sp.rand(L, M, N, density=1)

print t.shape
print t

idx = []
vs = []
for k in range(L):
    for j in range(N):
        for i in range(M):
            v = k * M * N + j * M + i
            vs.append((k, i, j, v))
            idx.append(v)
print vs
            
pts = []
for x in idx:
    LM = L*M
    LN = L*N
    MN = M*N
    
    j = x % LM
    i = 0
    k = 0
    
    #j = 
    pts.append((k, i, j))
print pts

m = rand(2, 3, density=1)

print m
