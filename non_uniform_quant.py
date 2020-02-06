# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:10:57 2020

@author: Lab716A-PC
"""

import numpy as np
import gensim.downloader as api
import gensim as gs
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from scipy import stats
from utils import*
import sys

#class Non_Uniform_Quantizer:
#    def __init__(self, rate, mu, s, c_scale):
#        self.rate = rate
#        self.mu = mu
#        self.s = s
#        self.c_sf= c_scale
#        self.labels = list(range(2**rate))
#        q = [(2*label + 1)/(2*(2**rate)) for label in self.labels]
#        self.y = self.c_sf * stats.norm.ppf(q, loc = mu, scale = s)
#        self.y = [float(x) for x in self.y]
#    
#    def encode(self, x):
#        n = len(x)
#        self.input = x
#        self.x_comp = np.zeros_like(x, dtype = np.int32)
#        for i in range(n):
#            self.x_comp[i] = np.argmin(abs(self.y - x[i]))
##            idx = np.searchsorted(self.y, x[i])
##            if idx == len(self.y):
##                self.x_comp[i] = idx - 1
##            else:
##                self.x_comp[i] = idx
#        
#        return self.x_comp
#
#class Non_Uniform_Dequantizer:
#    def __init__(self, rate, mu, s, c_scale):
#        self.rate = rate
#        self.mu = mu
#        self.s = s
#        self.c_sf= c_scale
#        self.labels = list(range(2**rate))
#        q = [(2*label + 1)/(2*(2**rate)) for label in self.labels]
#        self.y = self.c_sf * stats.norm.ppf(q, loc = mu, scale = s)
#        
#    def decode(self, x_comp):
#        n = len(x_comp)
#        self.input = x_comp
#        self.x_rxn = np.zeros_like(x_comp, dtype=np.float32)
#        for i in range(n):
#            self.x_rxn[i] = self.y[x_comp[i]]
#        return self.x_rxn

class Non_Uniform_Quantizer:
    def __init__(self, levels, mu, s, c_scale):
        self.rate = rate
        self.mu = mu
        self.s = s
        self.c_sf= c_scale
        self.labels = list(range(levels))
        q = [(2*label + 1)/(2*(levels)) for label in self.labels]
        self.y = self.c_sf * stats.norm.ppf(q, loc = mu, scale = s)
        self.y = [float(x) for x in self.y]
    
    def encode(self, x):
        n = len(x)
        self.input = x
        self.x_comp = np.zeros_like(x, dtype = np.int32)
        for i in range(n):
            self.x_comp[i] = np.argmin(abs(self.y - x[i]))
#            idx = np.searchsorted(self.y, x[i])
#            if idx == len(self.y):
#                self.x_comp[i] = idx - 1
#            else:
#                self.x_comp[i] = idx
        
        return self.x_comp

class Non_Uniform_Dequantizer:
    def __init__(self, levels, mu, s, c_scale):
        self.rate = rate
        self.mu = mu
        self.s = s
        self.c_sf= c_scale
        self.labels = list(range(levels))
        q = [(2*label + 1)/(2*(levels)) for label in self.labels]
        self.y = self.c_sf * stats.norm.ppf(q, loc = mu, scale = s)
        
    def decode(self, x_comp):
        n = len(x_comp)
        self.input = x_comp
        self.x_rxn = np.zeros_like(x_comp, dtype=np.float32)
        for i in range(n):
            self.x_rxn[i] = self.y[x_comp[i]]
        return self.x_rxn

def optScale(rate, mu, s, c_lb = 0.5, c_ub = 1.5, n_points = 50):
    """
    Optimize the scale factor of the reproducer values using a simple line search
    from c_lb to c_ub split into n_points
    """
    c = np.linspace(c_lb, c_ub, n_points)
    n_frames = 10
    n = 10000
    all_sqnr = []
    for c_val in c:
        x = np.random.normal(mu, s, n)
        quant = Non_Uniform_Quantizer(rate, mu, s, c_scale = c_val)
        dequant = Non_Uniform_Dequantizer(rate, mu, s, c_scale = c_val)
        quant.encode(x)
        #testing frames
        frame_sqnr = []
        for i in range(n_frames):
            x = np.random.normal(mu, s, n)
            x_comp = quant.encode(x)
            x_rxn = dequant.decode(x_comp)
            sqnr = 10*np.log10(s**2/np.mean((x - x_rxn)**2))
            frame_sqnr.append(sqnr)
        
        print("c_scale = {:.4}".format(c_val))
        print("mean SQNR = {:.4} {}".format(np.mean(frame_sqnr), "dB"))
        CI = stats.t.interval(0.95, len(frame_sqnr)-1, loc=np.mean(frame_sqnr), scale=stats.sem(frame_sqnr))
        print("95% CI size = {:.4} {}".format(CI[1] - CI[0], "dB")) 
        all_sqnr.append(np.mean(frame_sqnr))
        
    print("Best SQNR = {:.4} {}, c_scale = {}".format(np.max(all_sqnr), "dB",c[np.argmax(all_sqnr)] ))
    best_c = c[np.argmax(all_sqnr)]
    
    return best_c

def optScaleData(rate, data, c_lb = 0.5, c_ub = 1.5, n_points = 50):
    """
    Optimize the scale factor of the reproducer values using a simple line search
    from c_lb to c_ub split into n_points
    """
    c = np.linspace(c_lb, c_ub, n_points)
    mu, s = stats.norm.fit(data)
    all_sqnr = []
    for c_val in c:
        quant = Non_Uniform_Quantizer(rate, mu, s, c_scale = c_val)
        dequant = Non_Uniform_Dequantizer(rate, mu, s, c_scale = c_val)
        x_comp = quant.encode(data)
        x_rxn = dequant.decode(x_comp)
        sqnr_val = 10*np.log10(s**2/np.mean((data - x_rxn)**2))
        all_sqnr.append(sqnr_val)        
        print("c_scale = {:.4}".format(c_val))
        print("SQNR = {:.4} {}".format(sqnr_val, "dB"))
        
    print("Best SQNR = {:.4} {}, c_scale = {}".format(np.max(all_sqnr), "dB",c[np.argmax(all_sqnr)] ))
    best_c = c[np.argmax(all_sqnr)]
    
    return best_c

def compress(x, rate, c_scale):
    """
    Construct a per dimension quantizer to compress the input sequence x
    """
    n, d = x.shape
    x_comp = np.zeros((n, d), dtype=np.int32)
    for i in range(d):
        print("Compressing dimension = {}, bpi = {}".format(i, rate[i]))
        x_col = x[:, i]
        mu, s = stats.norm.fit(x_col)
#        quant = Non_Uniform_Quantizer(rate[i], mu, s, c_scale[rate[i]])
        quant = Non_Uniform_Quantizer(rate[i], mu, s, c_scale[i])
        encoded = np.array(quant.encode(x_col))
        x_comp[:, i] = encoded.T 
        
    return x_comp

def decompress(x_comp, rate, mu, s, c_scale):
    """
    Construct a per dimension reconstructor to decompress the quantized input 
    x_comp
    """
    n, d = x_comp.shape
    x_rxn = np.zeros((n, d), dtype=np.float32)
    for i in range(d):
        x_col = x_comp[:, i]
#        decoder = Non_Uniform_Dequantizer(rate[i], mu[i], s[i], c_scale[rate[i]])
        decoder = Non_Uniform_Dequantizer(rate[i], mu[i], s[i], c_scale[i])
        rxn = np.array(decoder.decode(x_col))
        x_rxn[:, i] = rxn.T
    
    return x_rxn


#n = int(1e4)
#mu = 0
#s = 0.1
#rate = 7
#c_sf = np.linspace(0.5, 5, 10)
#all_sqnr = []
#for c_val in c_sf:
#    data = np.random.normal(mu, s, n)
#    quant = Non_Uniform_Quantizer(rate, mu, s, c_val)
#    data_comp = quant.encode(data)
#    dequant = Non_Uniform_Dequantizer(rate, mu, s, c_val)
#    data_rxn = dequant.decode(data_comp)
#    sqnr = 10*np.log10(s**2/np.mean((data - data_rxn)**2))
#    all_sqnr.append(sqnr)
#    print("SQNR = {:.4} {}".format(np.mean(sqnr), "dB"))
#    
#c = c_sf[np.argmax(all_sqnr)]
#print("best c_sf = {}".format(c))
#
#plt.figure()
#plt.plot(data[:200])
#plt.plot(data_rxn[:200])

rates = list(range(8))
c_sf = {}
for rate in rates:
    c_best = optScale(rate, 0, 1, 0.5, 3, 30)
    c_sf[rate] = c_best



parser = argparse.ArgumentParser(description = 'Trellis quantizer')
parser.add_argument('--total_bits', type = int)
parser.add_argument('--word2vec_bin_file', type = str)
parser.add_argument('--vector_index_file', type = str)
assert len(sys.argv) != 1, 'no arguments provided' 
options = parser.parse_args()
#print(sys.argv)

wv = api.load('word2vec-google-news-300')

w = gs.models.KeyedVectors.load_word2vec_format(options.word2vec_bin_file, binary=True)

with open(options.vector_index_file) as fp:
    lines = fp.readlines()

idx = []
for line in lines:
    vals = line.split(',')
    idx.append(int(vals[0]))
    idx.append(int(vals[1]))
    
vectors = w.vectors[idx]

x, mean_vector, norm, eig, mag = preProcess(vectors, center = True, unit_norm = True, pca = True, toAngle = True)

n, d = x.shape
num_bins = 100
hist, be = np.histogram(mag, bins = num_bins, density = True)

mu, loc, scale = stats.invgauss.fit(mag)
invnorm_dist = stats.invgauss.pdf(be[:-1], mu, loc, scale)


phase_hist = []
phase_bin_edges = []
phase_fit_hist = []

for val in range(d):
    hist, be = np.histogram(x[:, val], bins = num_bins, density = True)
    phase_hist.append(hist)
    phase_bin_edges.append(be)
    mu, s = stats.norm.fit(x[:, val])
    norm_dist = stats.norm.pdf(be[:-1], mu, s)
    phase_fit_hist.append((mu, s))


dim_std = np.array(phase_fit_hist)[:,1]
min_dist_rates = rateAllocation(dim_std, 1e-3, 1e-1, 2000)
total_rates = np.sort(list(min_dist_rates.keys()))

#c_scale = np.load("./c_scale.npy", allow_pickle = True).item()


#bits = options.total_bits
#bits = 1800
#n, d = x.shape
#rate_idx = total_rates[np.searchsorted(total_rates, bits)] 
##total_rate = min_dist_rates[rate_idx]
#print("vectors = {}, dimensions = {}, total bits = {}".format(n, d, rate_idx)) 
#rate = np.array(min_dist_rates[rate_idx], dtype = np.int)

tgt_var = 1e-5
lvl_alloc = 0.5*np.log2(dim_std**2/tgt_var)
lvl_alloc = np.ceil(2**lvl_alloc)
lvl_alloc = np.array(lvl_alloc, dtype = np.int32)
per_dim_c_sf = 1.5*np.ones((d, ))
#per_dim_c_sf = []
#for i in range(d):
#    per_dim_c_sf.append(optScaleData(int(lvl_alloc[i]), x[:, i], 0.7, 2.5, 25))

x_comp = compress(x, lvl_alloc, per_dim_c_sf)
#np.savez("vec_comp_{}.npz".format(rate_idx), x_comp)
#np.savez("vec_mean_{}.npz".format(rate_idx), mean_vector)
#np.savez("vec_norm_{}.npz".format(rate_idx), norm)
#np.savez("eig_matrix_{}.npz".format(rate_idx), eig)
#np.savez("vec_mag_{}.npz".format(rate_idx), mag)

mu = np.array(phase_fit_hist)[:, 0]
s = np.array(phase_fit_hist)[:, 1]

x_rxn = decompress(x_comp, lvl_alloc, mu, s, per_dim_c_sf)

# visualize quantization and reconstruction
plt.figure()
plt.plot(x[:200, 0])
plt.plot(x_rxn[:200, 0])

#quant_error = np.abs((x - x_rxn)/x)

vectors_h = postProcess(x_rxn, mean_vector, norm, eig, mag)

true_sim = []
rxn_sim = []
for i in range(0, n, 2):
    v1 = vectors[i]
    v2 = vectors[i+1]
    cos_sim = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    true_sim.append(cos_sim)
    vh1 = vectors_h[i]
    vh2 = vectors_h[i+1]
    cos_sim_h = np.dot(vh1, vh2)/(np.linalg.norm(vh1)*np.linalg.norm(vh2))
    rxn_sim.append(cos_sim_h)

true_sim = np.array(true_sim)
rxn_sim = np.array(rxn_sim)

avg_error = np.mean(np.abs(true_sim - rxn_sim))
max_error = np.max(np.abs(true_sim - rxn_sim))
print("Avg absolute error = {:.4}".format(avg_error))
print("Max absolute error = {:.4}".format(max_error))
            

d = 299
hist = {}
max_val = np.max(x_comp[:, d])
for i in range(max_val + 1):
    hist[i] = 0
for val in x_comp[:, d]:
    hist[val] += 1
hist_vals = [hist[key] for key in hist.keys()]
plt.figure()
plt.stem(hist_vals)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

rate = 5
quant = Non_Uniform_Quantizer(rate, 0, 2, c_sf[rate])
plt.figure()
plt.stem(quant.y, quant.labels)


#hist, be = np.histogram(x_comp[:,0], bins = num_bins, density = True)
#plt.figure()
#plt.plot(be[:-1], hist)         

dim = 0
mean = mu[dim]
std = s[dim]
err = x - x_rxn
#err_mu = np.mean(err, axis = 0)
err_var = np.var(err, axis = 0)
mse = np.mean(err**2, axis = 0)
rel_err = err/x
rel_err_mu = np.mean(rel_err, axis = 0)
max_err = np.max(err[:, dim])
idx = np.argmax(err[:, dim])
true_val = x[idx, dim]
rxn_val = x_rxn[idx, dim]
max_err_y_vals = []
max_err_vals = []
avg_err_vals = []
x_vals = []
x_rxn_vals = []
err_mu = []
err_s = []
colwise_sqnr = 10*np.log10(s/err_s)
for i in range(d):
    err_mu.append(np.mean(err[:, i]))
    err_s.append(np.var(err[:, i]))
    idx = np.argmax(err[:, i])
    x_vals.append(x[idx, i])
    x_rxn_vals.append(x_rxn[idx, i])
    max_err_vals.append(x[idx, i] - x_rxn[idx, i])
    avg_err_vals.append(np.mean(err[:, i]))
    quant = Non_Uniform_Quantizer(rate[i], mu[i], s[i], c_sf[rate[i]])
    y = quant.y
    max_err_y_vals.append(np.searchsorted(y, x_rxn[idx, i]))


def cost(init_pts, i, j):
    rxn_val = np.mean(init_pts[i:j])
    cost = np.sum([(x - rxn_val)**2 for x in init_pts[i:j]])
    return cost


quant = Non_Uniform_Quantizer(10, 0, 1, 1)
init_pts = np.array(quant.y)

n = len(init_pts)
n_pts = 16
best_cost = np.zeros((n + 1, n_pts))

for j in range(1, n + 1):
    best_cost[j, 0] = cost(init_pts, 0, j)


for k in range(32, n_pts):
    for j in range(1, n + 1):
        min_cost = np.inf
        for i in range(j):
            best_cost_i_j = best_cost[i, k - 1] + cost(init_pts, i, j)  
            if best_cost_i_j < min_cost:
                min_cost = best_cost_i_j
        best_cost[j, k] = min_cost

row = n
path = []
for k in range(n_pts - 1, 0, -1):
    min_cost = np.inf
    for i in range(row):
        best_cost_i_j = best_cost[i, k - 1] + cost(init_pts, i, row)
        if best_cost_i_j < min_cost:
            min_cost = best_cost_i_j
            pt = (i, row)
    path.append(pt)
    row = pt[0]

path.append((0, path[-1][0]))

sel_idx = [tup[0] for tup in path]
rxn_vals = [np.mean(init_pts[tup[0]: tup[1]]) for tup in path]
rxn_vals = rxn_vals[::-1]
labels = list(range(len(rxn_vals)))
plt.figure()
plt.stem(labels, rxn_vals)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

quant_1 = Non_Uniform_Quantizer(3, mu[0], s[0], 1.68)
quant_2 = Non_Uniform_Quantizer(3, mu[0], s[0], 1.68)

dequant_1 = Non_Uniform_Dequantizer(3, mu[0], s[0], 1.68)
dequant_2 = Non_Uniform_Dequantizer(3, mu[0], s[0], 1.68)

quant_2.y = rxn_vals
dequant_2.y = rxn_vals

data = x[:, 0]

x_comp_1 = quant_1.encode(data)
x_comp_2 = quant_2.encode(data)

x_rxn_1 = dequant_1.decode(x_comp_1)
x_rxn_2 = dequant_2.decode(x_comp_2)

sqnr_1 = 10*np.log10(s[0]**2/np.mean((data - x_rxn_1)**2))
sqnr_2 = 10*np.log10(s[0]**2/np.mean((data - x_rxn_2)**2))
        
quant_ll = Non_Uniform_Quantizer(6, 0, 1, 1.674)
quant_ll.y = [-1.51, -0.45, 0.45, 1.51]
dequant_ll = Non_Uniform_Dequantizer(6, 0, 1, 1.674)
dequant_ll.y = [-1.51, -0.45, 0.45, 1.51]

data = np.random.normal(0, 1, int(5e5))
x_comp_ll = quant_ll.encode(data)
x_rxn_ll = dequant_ll.decode(x_comp_ll)
sqnr_ll = 10*np.log10(1**2/np.mean((data - x_rxn_ll)**2))
print(np.mean((data - x_rxn_ll)**2))
print(sqnr_ll)

h = 1e-3
val = 0
s_min = -10
s_max = 10
r = np.array(quant_ll.y)
u = []
for i in range(len(r) - 1):
    u.append((r[i] + r[i + 1])/2)
         
    
for i in range()
stats.norm.pdf(s + jh)*h

func = 0
s = s_min
while(s < s_max):
    rxn = r[np.argmin(abs(r - s))]
#    idx = np.searchsorted(r, s)
#    if idx < len(r):
#        rxn = r[idx]
#    else:
#        rxn = r[-1]
    func += (s - rxn)**2*stats.norm.pdf(s, loc = 0, scale = 1)*h
    s += h
    
print(func)
print(10*np.log10(1/func))




    


r = np.array(quant_ll.y)
n_pts = int(1e6)
num_bins = 200
data = np.random.normal(0, 1, n_pts)
iters = 100
for i in range(iters):
    print(r[0:len(r):10])
    mapping = {}
    data_cmp = []
#    u = []
#    for i in range(len(r) - 1):
#        u.append((r[i] + r[i + 1])/2)
    for j in range(len(r)):
        mapping[j] = []
    for val in data:
#        for j in range(len(u))
        idx = np.argmin(abs(r - val))
        data_cmp.append(idx)
        mapping[idx].append(val)
    
    data_cmp = np.array(data_cmp)

    new_r = []
    for idx in range(len(r)):
        interval_prob = len(mapping[idx])/n_pts
        interval_pts = mapping[idx]
        hist, be = np.histogram(interval_pts, bins = num_bins, density = True)
#        plt.figure()
#        plt.plot(be[:-1], hist)
        centroid = 0
        h = be[1] - be[0]
        for j in range(len(be) - 1):
            centroid += be[j]*hist[j]*h
        new_r.append(centroid)
    r = np.copy(new_r)
    

prob = 0
h = be[1] - be[0]
for i in range(len(be) - 1):
    prob += hist[i]*h

results = []
pts_set = [8, 16, 32, 64]
data = np.random.normal(0, 1, int(1e6))
test_data = np.random.normal(0, 1, int(1e6))
for k in pts_set:
    kmeans = KMeans(n_clusters = k, verbose = False)
    kmeans.fit(data[:, None])
    centers = kmeans.cluster_centers_
    quant_ll = Non_Uniform_Quantizer(int(np.log2(k)), 0, 1, 1)
    quant_ll.y = centers
    dequant_ll = Non_Uniform_Dequantizer(int(np.log2(k)), 0, 1, 1)
    dequant_ll.y = centers
    x_comp_ll = quant_ll.encode(test_data)
    x_rxn_ll = dequant_ll.decode(x_comp_ll)
    sqnr_ll = 10*np.log10(1**2/np.mean((test_data - x_rxn_ll)**2))
    print(np.mean((test_data - x_rxn_ll)**2))
    print(sqnr_ll)
    results.append(sqnr_ll)

for k in pts_set:
    c = optScale(int(np.log2(k)), 0, 1, 0.7, 2.2, 30)
    quant_ll = Non_Uniform_Quantizer(int(np.log2(k)), 0, 1, c)
    dequant_ll = Non_Uniform_Dequantizer(int(np.log2(k)), 0, 1, c)
    x_comp_ll = quant_ll.encode(test_data)
    x_rxn_ll = dequant_ll.decode(x_comp_ll)
    sqnr_ll = 10*np.log10(1**2/np.mean((test_data - x_rxn_ll)**2))
    print(np.mean((test_data - x_rxn_ll)**2))
    print(sqnr_ll)

for k_cl in pts_set:
    row = best_cost.shape[0] - 1
    path = []
    for k in range(k_cl - 1, 0, -1):
        min_cost = np.inf
        for i in range(row):
            best_cost_i_j = best_cost[i, k - 1] + cost(init_pts, i, row)
            if best_cost_i_j < min_cost:
                min_cost = best_cost_i_j
                pt = (i, row)
        path.append(pt)
        row = pt[0]
    
    path.append((0, path[-1][0]))
    
    sel_idx = [tup[0] for tup in path]
    rxn_vals = [np.mean(init_pts[tup[0]: tup[1]]) for tup in path]
    rxn_vals = rxn_vals[::-1]
    quant_ll = Non_Uniform_Quantizer(int(np.log2(k)), 0, 1, 1)
    quant_ll.y = rxn_vals
    dequant_ll = Non_Uniform_Dequantizer(int(np.log2(k)), 0, 1, 1)
    dequant_ll.y = rxn_vals
    x_comp_ll = quant_ll.encode(test_data)
    x_rxn_ll = dequant_ll.decode(x_comp_ll)
    sqnr_ll = 10*np.log10(1**2/np.mean((test_data - x_rxn_ll)**2))
    print(np.mean((test_data - x_rxn_ll)**2))
    print(sqnr_ll)