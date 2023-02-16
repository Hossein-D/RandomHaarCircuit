import numpy as np
import random as rand
from random import sample
from qiskit import QuantumCircuit
import qiskit as qiskit
import qiskit.visualization
from numpy import sqrt 
from numpy import transpose 
from numpy import conj
import scipy
from numpy import exp
from numpy import log
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
import csv
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit.visualization import plot_histogram

from qiskit import QuantumCircuit, transpile
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit import Aer
from qiskit import QuantumCircuit

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.test.reference_circuits import ReferenceCircuits
from qiskit_ibm_runtime import QiskitRuntimeService

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.providers.basicaer import QasmSimulatorPy

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from numpy import zeros
from numpy.random import rand
from numpy.linalg import qr
from numpy import linalg
from numpy.linalg import inv
from numpy.linalg import eig
from numpy.linalg import matrix_power

from numpy import matmul
from numpy import divide
from numpy import diagonal
from numpy import floor
from numpy import copy
from qiskit import quantum_info as qinfo

from qiskit.quantum_info import Statevector
from numpy import math
from numpy import pi
#service = QiskitRuntimeService()

#program_inputs = {'iterations': 1}
#options = {"backend_name": "ibmq_qasm_simulator"}
#job = service.run(program_id="hello-world",
#                options=options,
#                inputs=program_inputs
#                )
#print(f"job id: {job.job_id}")
#result = job.result()
#print(result)

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer

from numpy import log
from numpy import exp

import numpy as np
from numpy import *
from joblib import Parallel, delayed
from multiprocessing import Pool
import numpy as np
from time import clock     
#import qiskit fg
import matplotlib
import numpy as np
from random import randrange

import multiprocessing

import matplotlib.pyplot as plt
import sympy
from sympy import *
import itertools
from IPython.display import display
init_printing()
import math
from tempfile import TemporaryFile
#qiskit.__qiskit_version__
import numpy as np
import tensorflow as tf
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
import csv
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os, fnmatch
import sys, getopt

from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, SimpleRNN
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from scipy.optimize import minimize
from scipy.linalg import logm
from qiskit.compiler import assemble






def denMatCostDiag(lamSq, sx, sy, sz):
    legendre = lamSq[-1];
    mu = [[1/2*(1+sz), 1/2*(sx-1j*sy)], [1/2*(sx+1j*sy), 1/2*(1-sz)]]
    mueigval, mueigvec = eig(mu)
    #a=denVec[0]; b=denVec[1]; c=denVec[2]; d=denVec[3];
    print("mueigval = ", mueigval)
    Lambda = sum(np.multiply(np.multiply(lamSq[:-1], lamSq[:-1])-\
                             mueigval[:], np.multiply(lamSq[:-1], lamSq[:-1])-\
                             mueigval[:]))-legendre*(sum(np.multiply(lamSq[:-1], lamSq[:-1]))-1)
    
    return Lambda

def denMat(denVec):
    a=denVec[0]; b=denVec[1]; c=denVec[2]; d=denVec[3];
    rho = np.divide([[a**2+b**2+c**2, d*(b-1j*c)], [d*(b+1j*c), d**2]], \
                    (a**2+b**2+c**2+d**2))
    return rho






#def denMatCost(denVec, sx, sy, sz):
def denMatCost(denVec, sx, sy, sz):
    #print("denMat = ", denMat(denVec))
#    sx=-.1;sy=.5;sz=.5;
    a=denVec[0]; b=denVec[1]; c=denVec[2]; d=denVec[3];
    #print("denVec=", denVec)
    norm=(a**2+b**2+c**2+d**2);
    #cost = (1-a)**2+2*a
    #(2*b*d-sx)**2+(2*c*d-sy)**2
    cost1 = (2*b*d-norm*sx)**2/(norm*(2*b*d)) + \
    (2*c*d-norm*sy)**2/(norm*(2*c*d)) + \
    (a**2+b**2+c**2-d**2-norm*sz)**2/(norm*(a**2+b**2+c**2-d**2))
    #derivCost1da = (norm*sx-2*b*d)*(a*norm*sx+2*b*d*a)/(b*d*norm**2)+ \
    #(norm*sx-2*c*d)*(a*norm*sx+2*c*d*a)/(c*d*norm**2)+\
    #2*(2*a-2*a*sz)*(a**2+b**2+c**2-d**2-norm*sz)/(norm*(a**2+b**2+c**2-d**2))+\
    #-2*a*(a**2+b**2+c**2-d**2-norm*sz)**2/(norm**2*(a**2+b**2+c**2-d**2))+\
    #-(a**2+b**2+c**2-d**2-norm*sz)**2/(norm*(a**2+b**2+c**2-d**2)**2)
    
    
    cost2 = (2*b*d-norm*sx)**2/(norm**2*(sx)) + \
    (2*c*d-norm*sy)**2/(norm**2*sy) + \
    (a**2+b**2+c**2-d**2-norm*sz)**2/(norm**2*sz)
    #derivCost2da = 
    
    cost=cost1
    #derivCost=[derivCost1da, derivCost1db, derivCost1dc, derivCost1dd]
    
    #derivCost=[derivCost2da, derivCost2db, derivCost2dc, derivCost2dd]
    
    #print("cost = ", cost)
    #cost = (2*a**2+2*b**2+2*c**2-1-sz)**2
    #return cost1, cost2
    return cost

def denMatDerCost(denVec, sx, sy, sz):
    #print("denMat = ", denMat(denVec))
#    sx=-.1;sy=.5;sz=.5;
    a=denVec[0]; b=denVec[1]; c=denVec[2]; d=denVec[3];
    #print("denVec=", denVec)
    norm=(a**2+b**2+c**2+d**2);
    #cost = (1-a)**2+2*a
    #(2*b*d-sx)**2+(2*c*d-sy)**2
    derivCost1da=-2*a*sx*(2*b*d-norm*sx)/(b*d*norm) - a*(2*b*d-norm*sx)**2/(b*d*norm**2)\
    -2*a*sx*(2*c*d-norm*sx)/(c*d*norm) - a*(2*c*d-norm*sx)**2/(c*d*norm**2)\
    +2*(2*a-2*a*sz)*(a**2+b**2+c**2-d**2-norm*sz)/((a**2+b**2+c**2-d**2)*norm)\
    -2*a*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2)*(norm**2))\
    -2*a*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2)**2 * norm);
    
    derivCost1db=(2*d-2*b*sx)*(2*b*d-norm*sx)/(b*d*norm) - (2*b*d-norm*sx)**2/(d*norm**2)\
    -(2*b*d-norm*sx)**2/(2* b**2 *d*norm) - 2*b*sx*(2*c*d-norm*sx)/(c*d*norm)\
    -b*(2*c*d-norm*sx)**2/(c*d*norm**2)\
    +2*(2*b-2*b*sz)*(a**2+b**2+c**2-d**2-norm*sz)/((a**2+b**2+c**2-d**2)*(norm))\
    -2*b*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2) * norm**2)\
    -2*b*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2)**2 * norm);    

    derivCost1dc=-2*c*sx*(2*b*d-norm*sx)/(b*d*norm) - c*(2*b*d-norm*sx)**2/(b*d*norm**2)\
    +(2*d-2*c*sx)*(2*c*d-norm*sx)/(c*d*norm) - (2*c*d-norm*sx)**2/(d*norm**2)\
    -(2*c*d-norm*sx)**2/(2*c**2*d*norm)\
    +2*(2*c-2*c*sz)*(a**2+b**2+c**2-d**2-norm*sz)/((a**2+b**2+c**2-d**2)*(norm))\
    -2*c*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2) * norm**2)\
    -2*c*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2)**2 * norm);    

    derivCost1dd=(2*b-2*d*sx)*(2*b*d-norm*sx)/(b*d*norm) - (2*b*d-norm*sx)**2/(b*norm**2)\
    -(2*b*d-norm*sx)**2/(2*b*d**2*norm) + (2*c-2*d*sx)*(2*c*d-norm*sx)/(c*d*norm)\
    -(2*c*d-norm*sx)**2/(c*norm**2) - (2*c*d-norm*sx)**2/(2*c*d**2*norm)\
    +2*(-2*d-2*d*sz)*(a**2+b**2+c**2-d**2-norm*sz)/((a**2+b**2+c**2-d**2)*(norm))\
    -2*d*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2) * norm**2)\
    +2*d*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2)**2 * norm);    

    derivCost=[derivCost1da, derivCost1db, derivCost1dc, derivCost1dd]
        
    return derivCost

def maxLikelihoodDen(sx, sy, sz, methodML):
    Delta=1/4*(1-sx**2-sy**2-sz**2)
    M11=1/2*(1-sz);
    T0 =np.array([[sqrt(Delta/M11), 0], \
           [(sx+1j*sy)/(sqrt(2*(1-sz))), \
            sqrt(1/2*(1-sz))]])

    #denMat0=np.dot(np.transpose(conj(T0)), T0)
    #print("denMat0 = ", denMat0)
    #print("T0[1, 0] = ", np.real(complex(T0[1, 0])))
    #print("real T0[1, 0] = ", np.real(T0[1, 0]))
    denVec0=[np.real(complex(T0[0, 0])), np.real(complex(T0[1, 0])), \
             np.imag(complex(T0[1, 0])), np.real(complex(T0[1, 1]))]

    #a0=denMat0[0];    b0=denMat0[1];    c0=denMat0[2];    d0=denMat0[3];
    
    #denMatParam = [a, b, c, d];
    
    #res = minimize(fun, (2, 0), method='SLSQP') #, bounds=bnds,\
               #constraints=cons)
    #bnds = ((-float('inf'), float('inf')), (0, None))        
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #               method='SLSQP', jac=None)
    
    res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
                   method=methodML, jac=None)
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #               method='TNC', jac=None)

    
    physDenMat = denMat(res.x)
    
    #print("density mat from TNC = ", denMat(res.x))
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #method='Powell', jac=None)
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz),\
    #               method='Newton-CG', jac=denMatDerCost) #, hess=None, hessp=None) 
    #print("density mat from Newton = ", denMat(res.x))
    #print("res Newton = ", res)
    return physDenMat, res.x, res


def denMat3param(denVec):
    
    a=denVec[0]; b=denVec[1]; c=denVec[2];
    d=sqrt(1-a**2-b**2-c**2);
    rho = [[a**2+b**2+c**2, d*(b-1j*c)], [d*(b+1j*c), d**2]]
    
    return rho

def denMat3paramCost(denVec, sx, sy, sz):
    a=denVec[0]; b=denVec[1]; c=denVec[2]; 
    norm=1;d=sqrt(1-a**2-b**2-c**2); print("d=", d);
    cost1 = (2*b*d-norm*sx)**2/(norm*(2*b*d)) + \
    (2*c*d-norm*sy)**2/(norm*(2*c*d)) + \
    (a**2+b**2+c**2-d**2-norm*sz)**2/(norm*(a**2+b**2+c**2-d**2))
        
    cost2 = (2*b*d-norm*sx)**2/(norm**2*(sx)) + \
    (2*c*d-norm*sy)**2/(norm**2*sy) + \
    (a**2+b**2+c**2-d**2-norm*sz)**2/(norm**2*sz)
    
    cost=cost2
    
    #return cost1, cost2
    return cost


def maxLikelihoodDen3Param(sx, sy, sz):
    Delta=1/4*(1-sx**2-sy**2-sz**2)
    M11=1/2*(1-sz);
    T0 =np.array([[sqrt(Delta/M11), 0], \
           [(sx+1.0j*sy)/(sqrt(2*(1-sz))), \
            sqrt(1/2*(1-sz))]])
    print("T0=", T0)
    denMat0=np.matmul(np.transpose(conj(T0)), T0)
    print("denMat0 = ", denMat0)
    #print("T0[1, 0] = ", np.real(complex(T0[1, 0])))
    #print("real T0[1, 0] = ", np.real(T0[1, 0]))
    denVec0=[np.real(complex(T0[0, 0])), np.real(complex(T0[1, 0])), \
             np.imag(complex(T0[1, 0])), np.real(complex(T0[1, 1]))]
    
    #a0=denMat0[0];    b0=denMat0[1];    c0=denMat0[2];    d0=denMat0[3];
    
    #denMatParam = [a, b, c, d];
    
    #res = minimize(fun, (2, 0), method='SLSQP') #, bounds=bnds,\
               #constraints=cons)
    #bnds = ((-float('inf'), float('inf')), (0, None)) 
    print("denVec0 = ", denVec0)
    res = minimize(denMat3paramCost, denVec0[0:3], args=(sx, sy, sz), \
                   method='SLSQP', jac=None)
    print("density mat from SLSQP = ", denMat3param(res.x))
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #method='Powell', jac=None)
    #res = minimize(denMat3paramCost, denVec0, args=(sx, sy, sz),\
    #               method='Newton-CG', jac=denMatDerCost) #, hess=None, hessp=None) 
    #print("density mat from Newton = ", denMat(res.x))
    #print("res Newton = ", res)
    return res.x, res

#def maxLikelihoodDenDiag(sx, sy, sz):
#    return res.x, res
#A=[1, 2, 3]; A[0:2]; print("A = ", A)
#sx=0.5; sy=0.2; sz=0.9
#res = maxLikelihoodDen3Param(sx, sy, sz)
#T0=np.array([[0.707106781186548*1j, 0],
# [1.1180339887499 + 0.447213595499958*1j, 0.223606797749979]])
#print("T0 = ", np.matmul(T0, T0))
#A = np.array([[17.+0.j, -3.+0.j],
#              [-7.+0.j,  1.+0.j]])

#B = np.array([[ 60.+0.j,  -4.+0.j],
#              [-12.+0.j,   0.+0.j]])
#print("T0 = ", np.matmul(A, B))



def entanglement(rho):
    eigval = eig(rho)[0]
    EE = -np.sum(np.dot(np.log2(eigval), eigval))
    
    return EE




def HadamardMidQbit(circ, nqbit):
    circ.h(int(np.floor(nqbit-1)/2), nqbit-1)
    return circ
def CNOTWithLastQbit(circ, nqbit):
    circ.cx(int(np.floor(nqbit-1)/2), nqbit-1)
    return circ


def HaarGen(dim):
    # This function generates random Haar unitaries;
    
    A = rand(dim, dim)+1j*rand(dim, dim)
    A = A/np.sqrt(2)

    Q, R = linalg.qr(A)
    
    D = np.diag(np.diag(R))

    PhaseVec = divide(diagonal(D), abs(diagonal(D)))    
    
    PhaseArr = np.diag(PhaseVec)
    Rprime = matmul(inv(PhaseArr), R)
    Qprime = matmul(Q, PhaseArr)
    eigval, eigvec = eig(Qprime)
    
    QprimeCl1 = [[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1],[0, 0, 1, 0]] #np.identity(4) 
    QprimeCl2 = np.divide([[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]],sqrt(2))
    
    
    
    return Qprime, A, eigval


def angHaar(eigvalues):
    # This function calculates the angle of the eigenvalue vector. 
    ang = np.zeros(len(eigvalues), 1)
    ang = [np.angle(eigvalues[i]) for i in range(len(eigvalues))]
    
    return ang

def traceVecExceptLast(state, nqbit):
    # This function traces over all qubits except the last one. 
    stateTr1 = qinfo.partial_trace(state, np.arange(0, nqbit))

def statHaar(dim, Nsamp, Nbin):  
    min = -1;
    max = +1;
    binVec = LinRange(min, max, Nbin+1)
    deltaBin = (max-min)/Nbin    
    prob = zeros(Nbin)

    for n in range(Nsamp):
        if n%500==0:
            println("n = ", n)        
        Qprime, A, eig = HaarGen(dim)
        ang = angHaar(eig)
        for n in range(dim):
            numBin = Int(floor((ang[n]-min)/deltaBin))+1
            prob[numBin] = prob[numBin]+1        
    return prob
    


def genCircConfig(nqbit, circDepth, p):
    # This function generates the info for a hybrid random circuit which includes hte    
    # measurement locations and the random unitaries. 
    nNodes=nqbit*circDepth
    nUnitary=int(nNodes/2)
    measureVec=[(rand(1)[0]<p) for i in range(nNodes)]        
    
    measureArr=np.zeros((circDepth, nqbit))
    for t in range(circDepth):
        measureArr[t, :]=[measureVec[i+t*nqbit] for i in range(nqbit)]
    
    #print("measureArr = ", measureArr)
    unitaryArr=np.zeros((nUnitary, 4, 4))+0j    
    dim=4;
    for i in range(nUnitary):
        Q, A, eig = HaarGen(dim)
        unitaryArr[i, :, :] = np.copy(Q)
        #print("Q = ", np.real(Q))
        #print("Q*Q^{\dagger}=", matmul(Q, np.conj(np.transpose(Q))))
    return measureArr, unitaryArr



def timeEvolveAncilla(nqbit, circDep, p, initStateLabel, circConfig, renyiInd, refQbitAxis, Nshots):
    # This function evaluates the time evolution of the circuit when there is an ancilla qubit entangled
    # to the circuit. 
    
    nNodes=nqbit*circDep
    nUnitary=int(nNodes/2)
    measureArr=np.zeros((circDep, nqbit))
    measureVec = np.zeros(nNodes)
    # Here we generate the circuit in case it is already not generated. 
    
    if circConfig=="None":
        measureVec=[(rand(1)[0]<p)+0.0 for i in range(nNodes)]
        for t in range(circDep):
            measureArr[t, :]=[measureVec[i+t*nqbit] for i in range(nqbit)]

    # Here we retrieve the circuit configuration arrays in case it is already generated. 
    
    unitaryArr=np.zeros((nUnitary, 4, 4))+0j    
    dim=4;
    if circConfig!="None":
        measureArr=circConfig[0]
        unitaryArr=circConfig[1]        
        for t in range(circDep):        
            measureVec[t*nqbit:(t+1)*nqbit] = measureArr[t, :]
    
    print("measureArr = ", measureArr)
    print("measureVec = ", measureVec)
    
    # The ancilla qubit is put at the  of the string of the qubits. This way we tensor product the states
    # from left to right. The ancilla qbit is entangled to the qbit at the middle of the string at Ind=floor(nqbit/2)
    
    dim = 4;
    n2qbit = int(floor(nqbit/2))
    state = Statevector.from_int(0, 2**(nqbit+1))
    
    A=rand(2, 2)
    unitCirc = QuantumCircuit(nqbit+1, 2)
    midind = int(floor((nqbit-1)/2))
    # Add a H gate on qubit 0    
    #circuit.h(n-1)
    #circuit.cx(n-1, midind)

    #U = Operator(circuit)    
    #identity = Matrix(one(eltype(A))I, size(A,1), size(A,1))
    
    identityMat = np.identity(np.shape(A)[0])
    # In case we consider a mixed boundary condition.
    if initStateLabel=="mixed":
        for t in range(circDep):
            if t%2==1:
                #U = 1;
                for ngate in range(n2qbit):
                    Qprime, A, qprimEig = HaarGen(dim)                    
                    gate2x2 = unitCirc.unitary(Qprime, [2*ngate, 2*ngate+1])

            elif t%2==0:
                for ngate in range(n2qbit-1):
                    print("t = ", t)
                    Qprime, A, qprimEig = HaarGen(dim)
                    gate2x2 = unitCirc.unitary(Qprime, [2*ngate+1, 2*(ngate+1)])
                    
                

    state = state.evolve(unitCirc)
    
    # Here we create an entangled Bell pair between the ancilla and the qubit in the circuit. 
    
    BellGateCirc = QuantumCircuit(nqbit+1)
    BellGateCirc.initialize(state)
    
    hGate = BellGateCirc.h(midind)   # Hadamard gate on the last qubit. 
    cnotgate = BellGateCirc.cx(midind, nqbit)  # CNOT between the mid qubit and the last qubit. 
    state = state.evolve(BellGateCirc)
    #backend_sim = Aer.get_backend('qasm_simulator')

    # Execute the circuit on the qasm simulator.
    # We've set the number of repeats of the circuit
    # to be 1024, which is the default.
    #job = backend_sim.run(transpile(qc, backend_sim), shots=1024)
    
    qreg  = QuantumRegister((nqbit+1)) # 
    qregX  = QuantumRegister((nqbit+1)) # 
    qregY  = QuantumRegister((nqbit+1)) # 
    qregZ  = QuantumRegister((nqbit+1)) #     
    if refQbitAxis=="None": # In this case we measure the state of the reference qubit at the final step. 
        cr  = ClassicalRegister((nqbit)*circDep)
    else:
        cr  = ClassicalRegister((nqbit)*circDep+1)
        crX  = ClassicalRegister((nqbit)*circDep+1)
        crY  = ClassicalRegister((nqbit)*circDep+1)
        crZ  = ClassicalRegister((nqbit)*circDep+1)
        
    hybCirc = QuantumCircuit(qreg,cr)
    hybCircX = QuantumCircuit(qregX,crX)
    hybCircY = QuantumCircuit(qregY,crY)
    hybCircZ = QuantumCircuit(qregZ,crZ)   
    hybCirc.initialize(state)
    if refQbitAxis=="All":
        hybCircX.initialize(state)
        hybCircY.initialize(state)    
        hybCircZ.initialize(state)    
    Qprime=np.zeros((dim, dim))+0.0j
    for t in range(circDep):
        ### Measurement
        #cntMeasure = np.count_nonzero(measureArr)
        #qreg  = QuantumRegister(nqbit+1) 
        #cr  = ClassicalRegister(nqbit+1)
        if t%2==0:              
            for ngate in range(n2qbit):
                if circConfig=="None":                
                    Qprime, A, qprimEig = HaarGen(dim)
                else:
                    #unitInd=ngate+int(t/2)*nqbit
                    unitInd=ngate+t*n2qbit
                    Qprime=np.copy(unitaryArr[unitInd, :, :])
                #print("unitInd = ", unitInd)
                gate2x2=hybCirc.unitary(Qprime, [2*ngate, 2*ngate+1])
                if refQbitAxis=="All":                
                    gate2x2X=hybCircX.unitary(Qprime, [2*ngate, 2*ngate+1])
                    gate2x2Y=hybCircY.unitary(Qprime, [2*ngate, 2*ngate+1])
                    gate2x2Z=hybCircZ.unitary(Qprime, [2*ngate, 2*ngate+1])                
        elif t%2==1:            
            for ngate in range(1, n2qbit+1):    
                if circConfig=="None":
                    Qprime, A, qprimEig = HaarGen(dim)
                else:
                    unitInd=ngate-1+t*n2qbit
                    
                if ngate!=n2qbit:                    
                    gate2x2 = hybCirc.unitary(Qprime, [2*ngate-1, 2*ngate])
                    if refQbitAxis=="All":                                    
                        gate2x2X = hybCirc.unitary(Qprime, [2*ngate-1, 2*ngate])                    
                        gate2x2Y = hybCirc.unitary(Qprime, [2*ngate-1, 2*ngate])                    
                        gate2x2Z = hybCirc.unitary(Qprime, [2*ngate-1, 2*ngate])                                        
                else: 
                    gate2x2 = hybCirc.unitary(Qprime, [2*ngate-1, 0])                    
                    if refQbitAxis=="All":                                    
                        gate2x2X = hybCirc.unitary(Qprime, [2*ngate-1, 0])                    
                        gate2x2Y = hybCirc.unitary(Qprime, [2*ngate-1, 0])                    
                        gate2x2Z = hybCirc.unitary(Qprime, [2*ngate-1, 0])                                        
                    
                    
        for m in range(nqbit):
            if measureArr[t, m]:
                hybCirc.measure(qreg[m], cr[m+nqbit*t])                
                if refQbitAxis=="All":                                                    
                    hybCircX.measure(qregX[m], crX[m+nqbit*t])
                    hybCircY.measure(qregY[m], crY[m+nqbit*t])
                    hybCircZ.measure(qregZ[m], crZ[m+nqbit*t])
    if refQbitAxis=="Z":
        hybCirc.measure(qreg[nqbit], cr[(nqbit)*circDep])
    elif refQbitAxis=="X":
        hybCirc.h(nqbit)
        hybCirc.measure(qreg[nqbit], cr[(nqbit)*circDep])
    elif refQbitAxis=="Y":
        hybCirc.u(np.pi/2, np.pi/2, np.pi, nqbit);
        hybCirc.measure(qreg[nqbit], cr[(nqbit)*circDep])
    elif refQbitAxis=="All":
        hybCircZ.measure(qregZ[nqbit], crZ[(nqbit)*circDep])                
        hybCircX.measure(qregX[nqbit], crX[(nqbit)*circDep])        
        hybCircY.u(np.pi/2, np.pi/2, np.pi, nqbit);
        hybCircY.measure(qregY[nqbit], crY[(nqbit)*circDep])

        
    #backend_sim = Aer.get_backend('qasm_simulator')
    backend_sim = Aer.get_backend('statevector_simulator')
    
        # Execute the circuit on the qasm simulator.
        # We've set the number of repeats of the circuit
        # to be 1024, which is the default.
    if refQbitAxis=="None":
        job = backend_sim.run(transpile(hybCirc, backend_sim), shots=1)

    elif refQbitAxis=="All":        
        jobX = backend_sim.run(transpile(hybCircX, backend_sim), shots=Nshots)
        jobY = backend_sim.run(transpile(hybCircY, backend_sim), shots=Nshots)
        jobZ = backend_sim.run(transpile(hybCircZ, backend_sim), shots=Nshots)                
    else:
        job = backend_sim.run(transpile(hybCirc, backend_sim), shots=Nshots)        
        my_qobj = assemble(hybCirc)
        #my_qobj = assemble(c)
        #result = simulator.run(my_qobj).result()
        
        #backend = BasicAer.get_backend('statevector_simulator')
        #job = backend.run(transpile(qc, backend))
        #job=qiskit.execute(qc,backend,shots=500)
    #my_qobj = assemble(c)
    #result = simulator.run(my_qobj).result()
    
    counts = job.result().get_counts(hybCirc)
    if refQbitAxis=="All":            
        countsX = job.result().get_counts(hybCircX)
        countsY = job.result().get_counts(hybCircY)
        countsZ = job.result().get_counts(hybCircZ)
    
    convCounts = {}
    convKVec = []
    
    if refQbitAxis!="None" and refQbitAxis!="All":
        print("counts = ", counts)
        for k,v in counts.items():
            #print("k[::-1] = ", k[::-1])
            #print("v = ", v)
            
            
            # k inverse of measureVec
            # tempInvK inverse of k => tempInvK aligned with measureVec
            tempInvK = k[::-1]                             
            tempInvKArr = list(tempInvK)
            tempInvKConvArr = [int(i) for i in tempInvKArr]            
            convK = 2*np.multiply(tempInvKConvArr[:-1], measureVec) - measureVec
            #convK = -1*np.multiply(tempInvKConvArr[:-1], measureVec) + 2*measureVec

            if tempInvKConvArr[-1]==0:
                convK=np.append(convK, [0])
                #convK=np.append(convK, [2])
            elif tempInvKConvArr[-1]==1:
                convK=np.append(convK, [1])
            #print("convK = ", convK)
            for repetition in range(v):
                convKVec = np.append(convKVec, convK, axis=0)            
            convKStr = ''.join(str(int(x)) for x in convK)
            convCounts.update({convKStr: v})
        print("\n")
        return measureArr, convKVec, counts
    
            
    if refQbitAxis=="None":
        finalState = job.result().get_statevector(hybCirc)
        finStVec = finalState.data;
        finStVec = finStVec.reshape(len(finStVec), 1)
        finalDMArr = matmul(finStVec, conj(transpose(finStVec)))
        rhoFinDM = qinfo.DensityMatrix(finalDMArr)
    
        traceVec = np.arange(0, nqbit)
        traceState = rhoFinDM.copy()
        for i in range(nqbit):
            traceState = qinfo.partial_trace(traceState, [0])    
        
        rhoAncilla = np.copy(traceState);
        eigRho, eigVecRho = eig(rhoAncilla)
        sigmax=[[0, 1],[1, 0]]; sigmay=[[0, -1j],[1j, 0]]; sigmaz=[[1, 0],[0, -1]]; 
        rx = np.trace(matmul(rhoAncilla, sigmax))
        ry = np.trace(matmul(rhoAncilla, sigmay))
        rz = np.trace(matmul(rhoAncilla, sigmaz))    
        rvec=[rx, ry, rz];
        rvecNorm = np.linalg.norm(rvec)
    
        eps = (1+1j)*1e-16;
        eigvalsDen = [1/2-rvecNorm/2, 1/2+rvecNorm/2];

        if renyiInd==1:
            try:
                renyiEnt = -np.real(sum([eigvalsDen[i]*np.log2((eigvalsDen[i])) for i in range(2)]))
            except y:
                if isa(y, DomainError):
                    println("domainError")
                    println("eigvalsDen = ", eigvalsDen)
                    println("ancillaDenMat = ", ancillaDenMat)                    
        else:
            renyiEnt = 1/(1-renyiInd) * np.log2(np.trace(matrix_power(rhoAncilla, renyiInd)))
            renyiEnt = np.real(renyiEnt)
    
        print("renyiEnt = ", renyiEnt)
        
        return state, rhoAncilla, renyiEnt, measureArr    
