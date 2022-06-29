# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:27:41 2021

@author: suhai
"""

import numpy as np
import torch
import datetime
from torch import nn
import scipy.io
#import h5py
import torch.utils.data as Data




########## classes and functions definitions ################
def generate_traj(x0,u):
    q=[]
    for index, u_t in u:
        q.append(q0+u_t)
    return q

class Feedforward(nn.Module):
    def __init__(self):
        super(Feedforward, self).__init__()
     
        self.input_size = INPUT_SIZE
        self.hidden_size  = HIDDEN_SIZE
        self.fc1 = torch.nn.Linear(self.INPUT_SIZE, self.HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.HIDDEN_SIZE, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):    
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


def tansig(x,u_min,u_max):
    return u_min + (u_max-u_min)*(torch.tanh(x)+1)/2

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(       
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=NUM_LAYERS,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout = 0.5,
#            nonlinearity = 'relu'
        )
        
        self.out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x, h_state):    
        # x (batch, time_step, input_size)
        # r_out (batch, time_step, hidden_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(tansig(self.out(r_out[:, time_step, :]),torch.tensor([-MAX_Ctrl,-MAX_Ctrl,-MAX_Ctrl,-MAX_Ctrl]),torch.tensor([MAX_Ctrl,MAX_Ctrl,MAX_Ctrl,MAX_Ctrl])))
        return torch.stack(outs, dim=1), h_state
    
    
##########  Model learning #############
# Generate trajectories 



# approximate the model






##########  Policy learning #############


data= scipy.io.loadmat('DATA1.mat') 
Types = [11,11,10,0,0,10,0]

X = data.get('X')
Y = data.get('Y')
#Types = data.get('Types')

X = np.array(X)
Y = np.array(Y)
print(X[0,:,0])
print(Y[0,:,0])
print(X.shape)
print(Y.shape)

# parameters of the dataset (check them carefully)

nCtrb      = 2
nAgents    = len(Types)
TEST_START = 800
MAX_Ctrl   = np.max(Y)
MAX_Ctrl   = MAX_Ctrl.item()
n          = 50          # number of testpoints after training 

# Hyper Parameters for the NN

EPOCH       = 500
BATCH_SIZE  = 200
TIME_STEP   = 15
INPUT_SIZE  = 2*nAgents    # 
OUTPUT_SIZE = 2*nCtrb
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
LR          = 0.01 # learning rate

U   = torch.from_numpy(Y[:TEST_START,:,0:TIME_STEP]).float()
Q   = torch.from_numpy(X[:TEST_START,0:INPUT_SIZE,0:TIME_STEP]).float()
U_t = torch.from_numpy(Y[TEST_START:TEST_START+100,:,0:TIME_STEP]).float()
Q_t = torch.from_numpy(X[TEST_START:TEST_START+100,0:INPUT_SIZE,0:TIME_STEP]).float()

U = torch.transpose(U,1,2)
Q = torch.transpose(Q,1,2)
U_t = torch.transpose(U_t,1,2)
Q_t = torch.transpose(Q_t,1,2)

print(U.shape,Q.shape,U_t.shape,Q_t.shape)

Dataset = Data.TensorDataset(Q, U)

loader = Data.DataLoader(
    dataset=Dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               
)


    
rnn = RNN()
print(rnn)
print(rnn.parameters())

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  
loss_func = nn.MSELoss()                       
loss_best = 100
print(optimizer.param_groups[0]['lr'])
start = datetime.datetime.now()

#for step, (b_x, b_y) in enumerate(loader):
#print (b_x.shape, b_y.shape)
    
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(loader):        # gives batch data

        output, h_state = rnn(b_x,None)                               # rnn output
        loss = loss_func(output, b_y)                 
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients
        
        rnn.eval()
        test_output, h_state_t = rnn(Q_t,None)                  
        loss_t = loss_func(test_output, U_t)
        rnn.train()
        if step % 10 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test loss: %.4f' % loss_t.data.numpy())
        if loss_t.data.numpy() < loss_best:
            loss_best = loss_t.data.numpy()
        rnn_name = 'rnn_11_01.pkl'
        torch.save(rnn, rnn_name)
            #print('test loss: %.4f' % loss_t.data.numpy(), 'Save!')
end = datetime.datetime.now()
print('Training time:')
print(end-start)

#q_tensor = torch.from_numpy(X[0,0:INPUT_SIZE,1]).float().view(1,1,INPUT_SIZE)
#u_tensor, h_state_n = rnn(q_tensor,None)

print('Training has finished')
#print(q_tensor)
#print(u_tensor)
#print(h_state_n)

########################### testing  #############

print('Starting testing')
import matlab.engine

eng = matlab.engine.start_matlab()
#tf = eng.isprime(37) 
#print(tf) # check if matlab works

rnn = torch.load(rnn_name)
rnn.eval()

ro_sum  = 0
nb_suc  = 0
start   = datetime.datetime.now()
ref_sum = datetime.datetime.now()-datetime.datetime.now()
ref_num = 0
Q       = np.empty((0, OUTPUT_SIZE, 16))


for i in range(n):
    print(i)
    test_point = TEST_START-300+i
    #print(test_point)
    q0 = X[test_point,0:INPUT_SIZE,0]
    q = np.zeros((INPUT_SIZE,TIME_STEP+1))
    u = np.zeros((OUTPUT_SIZE,TIME_STEP))

    q[:,0] = q0
    h_state = None
    
    for j in range(TIME_STEP):
        start0 = datetime.datetime.now()
        q_tensor = torch.from_numpy(q[:,j]).float().view(1,1,INPUT_SIZE)
        u_tensor, h_state_n = rnn(q_tensor, h_state)
        u_ref = u_tensor.detach().numpy().reshape(-1)
        h_state = h_state_n
        end0    = datetime.datetime.now()
        ref_sum = ref_sum + end0-start0
        ref_num = ref_num + 1        
        q_ref   = q_tensor.detach().numpy().reshape(-1)
        u_mat   = matlab.double(u_ref.tolist())
        q_mat   = matlab.double(q_ref[0:2*nAgents].tolist())
        q_temp  = np.asarray(eng.getIn2(u_mat,q_mat,nCtrb,nAgents))
        q[:,j+1]= q_temp[0:INPUT_SIZE].reshape((INPUT_SIZE,))
        u[:,j]  = u_ref #reference control
    
    uall_mat = matlab.double(u.tolist())
    uall_opt = matlab.double(Y[test_point,:,:].tolist())
    q0_mat    = matlab.double(q0[0:2*nAgents].tolist())
    Types_mat = matlab.double(Types)
    ro = np.double(eng.objectiveFun(uall_mat,q0_mat,Types_mat,nCtrb))
    ro_opt = np.double(eng.objectiveFun(uall_opt,q0_mat,Types_mat,nCtrb))
    # Q = np.concatenate((Q,q.reshape(1,3,-1)),axis = 0)
    
    print(ro)
    print(ro_opt)
    ro_sum = ro_sum + ro
    grid_min = -2.5 
    grid_max = 2.5 
    grid = matlab.double( [grid_min ,grid_max])
    squary =matlab.double([1.9,1.9,0.6,0.6
             -2.4,-2.4,0.6,0.6]);
    
    # print(uall_mat)
    if ro > 0.005:
        nb_suc = nb_suc+1
        if (nb_suc % 2)  == 0:
            q1 = q0_mat
            u1 = uall_mat
            u11 = uall_opt
            #
        if (nb_suc%3) == 0:
            q2 = q0_mat
            u2 = uall_mat
            u22 = uall_opt
            #
        if (nb_suc%5) == 0:
            q3 = q0_mat
            u3 = uall_mat
            u33 = uall_opt
            #eng.plot_ctrl(u11,q1,Types_mat,grid,squary,nCtrb,nAgents,11,nargout=0)
            #eng.plot_ctrl(u22,q2,Types_mat,grid,squary,nCtrb,nAgents,22,nargout=0)
            #eng.plot_ctrl(u33,q3,Types_mat,grid,squary,nCtrb,nAgents,33,nargout=0)
            eng.plot_paths(u1,u2,u3,u11,u22,u33,q1,q2,q3,Types_mat,grid,nCtrb,nAgents,nb_suc-2,nargout=0)

        #eng.plot_ctrl(uall_mat,q0_mat,Types_mat,grid,squary,nCtrb,nAgents,nb_suc,nargout=0)
        #eng.plot_ctrl(uall_opt,q0_mat,Types_mat,grid,squary,nCtrb,nAgents,nb_suc+n,nargout=0)
    if ro > 0.005:
        nb_suc = nb_suc+1
        if (nb_suc % 2)  == 0:
            q1 = q0_mat
            u1 = uall_mat
            u11 = uall_opt
            #
        if (nb_suc%5) == 0:
            q2 = q0_mat
            u2 = uall_mat
            u22 = uall_opt
            #
        if (nb_suc%7) == 0:
            q3 = q0_mat
            u3 = uall_mat
            u33 = uall_opt
            #eng.plot_ctrl(u11,q1,Types_mat,grid,squary,nCtrb,nAgents,11,nargout=0)
            #eng.plot_ctrl(u22,q2,Types_mat,grid,squary,nCtrb,nAgents,22,nargout=0)
            #eng.plot_ctrl(u33,q3,Types_mat,grid,squary,nCtrb,nAgents,33,nargout=0)
            eng.plot_paths(u1,u2,u3,u11,u22,u33,q1,q2,q3,Types_mat,grid,nCtrb,nAgents,nb_suc-2,nargout=0)

        #eng.plot_ctrl(uall_mat,q0_mat,Types_mat,grid,squary,nCtrb,nAgents,nb_suc,nargout=0)
        #eng.plot_ctrl(uall_opt,q0_mat,Types_mat,grid,squary,nCtrb,nAgents,nb_suc+n,nargout=0)
end = datetime.datetime.now()
print('average time: ')
print((end-start)/n)
print('success rate:')
print(nb_suc/n)
print('average robustness:')
print(ro_sum/n)