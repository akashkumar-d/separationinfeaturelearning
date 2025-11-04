import numpy as np

import torch
import torch.nn as nn
import math
from math import sqrt
from torch.func import functional_call, vmap, vjp, jvp, jacrev
# import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.optim as optim

from scipy.linalg import eigh
import time

import pickle
import copy

n = 0
net = None
def fnet_single(params, x):
    return functional_call(net, params, (x.unsqueeze(0),)).squeeze(0)

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = jac1
    # jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    # jac2 = jac2.values()
    # jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

def Kspectrum(X, net, fnet_single, s=1, only_eigvals=True):
    params = {k: v.detach() for k, v in net.named_parameters()}
    
    result = empirical_ntk_jacobian_contraction(fnet_single, params, X, X)
    result = result.detach().numpy().reshape(n,n)

    if only_eigvals:
        lams, _ = eigh(result, subset_by_index=[n-s, n-1])
        return lams
    else:
        lams, vecs = eigh(result, subset_by_index=[n-s, n-1])
        return lams, vecs
        
def top_eigs(K, s=3):
    lams, vecs = eigh(K, subset_by_index=[n-s, n-1])
    return lams, vecs
    
class Net(nn.Module):
    def __init__(self, d, w, bias):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d, w, bias=bias)
        self.fc2 = nn.Linear(w, 1, bias=bias)
        self.d = d
        self.w = w
        self.bias = bias
        self.initialize()

    def forward(self, x):
      if self.bias:
        x = self.fc1(x) / sqrt(self.d) + (1 - 1 / sqrt(self.d) ) * self.fc1.bias
        x = torch.relu(x) 
        x = self.fc2(x) / sqrt(self.w) + (1 - 1 / sqrt(self.w) ) * self.fc2.bias
        return x
      else:
        x = torch.relu(self.fc1(x) / sqrt(self.d)) 
        x = self.fc2(x) / sqrt(self.w)
        return x

    def initialize(self):
        self.fc1.weight.data.normal_(0, 1)
        self.fc2.weight.data.normal_(0, 1)
        if self.bias:
          self.fc1.bias.data.normal_(0, 1)
          self.fc2.bias.data.normal_(0, 1)
            
    def small_init(self):
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        if self.bias:
          self.fc1.bias.data.normal_(0, 0.1)
          self.fc2.bias.data.normal_(0, 0.1)

# def train(seed, net, dataset, epochs, lr0 =  , milestones, lrs = , track_spectrum = False):
# net = Net(100, 1024, bias=False)

def train( seed, X, Y, net_, EGOP, epochs = 100, lr0 = 2, num_eigs=3, lrs = [8, 16, 30, 50, 60, 80], milestones = [50, 150, 220, 280, 350, 400], exp_data=None, track_spectrum = False):
    torch.manual_seed(seed)
    global n
    global net 
    n = Y.shape[0]
    net = net_
    losses = []
    ## Journaling variables
    GradNorms = []
    ParamNorms = []
    Params = []
    # Ms = []
    Lams = []
    Eigvecs = []
    Eigvals = []
    Zs = []
    aligns = []
    Outputs = []
    Ks = []
    ## create a dict to track data by-reference
    return_flag = False
    if exp_data is None:
        return_flag = True
        exp_data = {}
        
    keyvals = {
        'losses' : losses,
        'aligns' : aligns,
        'ParamNorms': ParamNorms,
        'GradNorms' : GradNorms,
        'Params' : Params,
        'net' : copy.deepcopy(net),
        'Lams' : Lams,
        'Zs' : Zs,
        'Outputs' : Outputs,
        'Eigvecs' : Eigvecs,
        'Eigvals' : Eigvals,
        'Ks' : Ks,
    }
    for k, v in keyvals.items():
        exp_data[k] = v

    ## training
    criterion = torch.nn.MSELoss()
    idx = 0
    eta = lr0

    ## get initial values
    bunch0 = []
    bunch1 = []
    for param in net.parameters():      
        s = param.data
        nor = torch.norm(s)
        bunch0.append(s)
        bunch1.append(nor)
    Params.append(bunch0)
    ParamNorms.append(bunch1)
    GradNorms.append([0,0])

    if track_spectrum:
        params = {k: v.detach() for k, v in net.named_parameters()}
        K = empirical_ntk_jacobian_contraction(fnet_single, params, X, X) 
        K = K.detach().numpy().reshape(n,n)
        vals, vecs = top_eigs(K, s=num_eigs)
        Eigvals.append(vals) ; Eigvecs.append(vecs)
        Ks.append(K)
        
    pred = net(X)
    Zs.append( (pred-Y).data )
    loss = criterion(pred, Y)
    losses.append(loss.item())

    ## first alignment measurment
    X.requires_grad = True
    jac = vmap(jacrev(net))(X)
    M = torch.einsum('ijk,ijl->ikl', jac, jac).mean(axis=0)
    M = M.detach().cpu().numpy()
    X.requires_grad = False
    
    algn = np.trace(M.T.dot(EGOP) ) / ( np.sqrt( np.sum(M**2) ) * np.sqrt( np.sum(EGOP**2) )  + 1e-10 )
    aligns.append(algn)   

        
    ## Start training
    for epoch in range(epochs):
        net.zero_grad()
        pred = net(X)
        Outputs.append(pred.detach().numpy())
        loss = criterion(pred, Y)
        loss.backward()
        z = (pred - Y).data
        Zs.append(z)
        losses.append(loss.item())
        
        if epoch in milestones:
            lam = Kspectrum(X, net, fnet_single).item()
            Lams.append(lam)
            eta = lrs[idx]
            idx += 1
        
        bunch0 = []
        bunch1 = []
        bunch2 = []
        for param in net.parameters():      
            param.data = param.data - eta * param.grad.data
            s = param.data
            bunch0.append(s)
            s = torch.norm(param.data)
            bunch1.append(s)
            s = torch.norm(param.grad.data)
            bunch2.append(s)
        Params.append(bunch0)
        ParamNorms.append(bunch1)
        GradNorms.append(bunch2)
        net.zero_grad()
        
        # measure AGOP alignment
        X.requires_grad = True
        jac = vmap(jacrev(net))(X)
        M = torch.einsum('ijk,ijl->ikl', jac, jac).mean(axis=0)
        M = M.detach().cpu().numpy()
        X.requires_grad = False
        
        algn = np.trace(M.T.dot(EGOP) ) / ( np.sqrt( np.sum(M**2) ) * np.sqrt( np.sum(EGOP**2) )  + 1e-10 )
        aligns.append(algn)     
        
        if track_spectrum:
            params = {k: v.detach() for k, v in net.named_parameters()}
            K = empirical_ntk_jacobian_contraction(fnet_single, params, X, X) 
            K = K.detach().numpy().reshape(n,n)
            Ks.append(K)
            vals, vecs = top_eigs(K, s=num_eigs)
            # vals, vecs = Kspectrum(X, net, fnet_single, s=3, only_eigvals=False)
            Eigvals.append(vals) ; Eigvecs.append(vecs)

        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, AGOP Align.: {algn:.4f}')

    if return_flag:
        return exp_data

def get_NTK_characteristics(net, params, s=4):
    net = copy.deepcopy(net)
    with torch.no_grad():
        for param, val in zip(net.parameters(),params):
            param.copy_(val, )
    #
    # (X, net, fnet_single, s=1, only_eigvals=True)
    # Kspectrum(X, net, fnet_single, s=1, only_eigvals=True)
def mat_cos(A, B):
    frob_norm = lambda x : np.sqrt( np.sum(x**2) )
    return np.trace( A.T @ B) / (frob_norm(A)*frob_norm(B) + 1e-10)

def get_AGOP(net, X):
    X.requires_grad = True
    jac = vmap(jacrev(net))(X)
    M = torch.einsum('ijk,ijl->ikl', jac, jac).mean(axis=0)
    M = M.detach().cpu().numpy()
    X.requires_grad = False
    return M

def get_NTK(net, X):
    params = {k: v.detach() for k, v in net.named_parameters()}
    def fnet_single(params, x):
        return functional_call(net, params, (x.unsqueeze(0),)).squeeze(0)
    K = empirical_ntk_jacobian_contraction(fnet_single, params, X, X)
    return K.squeeze().numpy()
