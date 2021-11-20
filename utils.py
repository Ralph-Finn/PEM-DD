import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from scipy.stats import skew
use_gpu = torch.cuda.is_available()
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import cycle, islice
from sklearn.decomposition import NMF
from matplotlib.patches import Ellipse
from  sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import axes3d
from scipy import stats
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

# PE function
def pt(x,s=3,t=2):
    p = 1/torch.pow(x,s)-1/torch.pow(x,t)
#     p = 1/torch.pow(x,0.5)
    return p

def split_dataset(data,labels,num):
    skf = StratifiedKFold(n_splits=num)
    train_index, test_index = next(iter(skf.split(data, labels)))
    X_train = data[train_index]
    y_train = labels[train_index]
    X_test = data[test_index]
    y_test = labels[test_index]
    return X_train,X_test,y_train,y_test

# def valide(X,base_means,model):
#     fs = torch.tensor(X,dtype = torch.float32)
#     fs = model(fs)
#     fs = fs.detach().numpy()
#     fx = torch.tensor(base_means,dtype = torch.float32)
#     fx = model(fx)
#     fx = fx.detach().numpy()
#     return fs,fx

def logistic_test(X_train,y_train,X_test,y_test):
    classifier = LogisticRegression(max_iter=1000).fit(X=X_train, y=y_train)
    pre = classifier.predict(X_test)
    idx = np.where(pre == y_test)[0]
    acc = idx.shape[0]/y_test.shape[0]
    return acc
    
# def near_test(base_means,X_test,y_test):
#     x = X_test
#     labels = np.empty((x.shape[0],))
#     dists = np.empty((0, base_means.shape[0]))
#     dist = np.zeros(base_means.shape[0])
#     for i, sample in enumerate(x):
#         dist = np.sum(np.multiply(sample - base_means, sample - base_means), (1))
#         labels[i] = np.argmin(dist)
#     idx = np.where(labels == y_test)[0]
#     acc = idx.shape[0]/labels.shape[0]
#     return acc

def knn_test(X_train,y_train,X_test,y_test):
    classifier = KNeighborsClassifier(n_neighbors=5).fit(X=X_train, y=y_train)
    pre = classifier.predict(X_test)
    idx = np.where(pre == y_test)[0]
    acc = idx.shape[0]/y_test.shape[0]
    return acc

# def near_res(base_means,X_test,y_test):
#     x = X_test
#     labels = np.empty((x.shape[0],))
#     dists = np.empty((0, base_means.shape[0]))
#     dist = np.zeros(base_means.shape[0])
#     for i, sample in enumerate(x):
#         dist = np.sum(np.multiply(sample - base_means, sample - base_means), (1))
#         labels[i] = np.argmin(dist)
#     return labels

class MCC(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim
    ):
        super(MCC, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = F.dropout(x,p=0.2)
        x = self.fc1(x)
        return x
    
# class PTMLoss(nn.Module):
#     def forward(self, x):
# #         S = torch.sum(x,dim=1)
# #         ls = torch.ones(x.shape[0])
# #         Q = x@x.t()
#         loss =  torch.mean(torch.clamp(ptm(0.1*F.pdist(x)/x.shape[1]+0.05),-1,1000))   #使用斥力算法让中心点远离,在距离上稍微加上一点，防止数据无法拟合
# #         s1 = torch.mm(x,x.t())
# #         print(s1.shape)
# #         loss = torch.sum(s1)
# #         loss =  torch.mean(F.pdist(x))
#         return loss

# class PairLoss(nn.Module):
#     def forward(self,x,y,dx):
# #         centers = dx
#         dists = 0
#         for i in range(dx.shape[0]):
#             mask = y == i
#             cluster_samples = x[mask]
# #             center = centers[i]
#             center = torch.mean(cluster_samples,dim =0)
# #             center = dx[i]
#             s1 = torch.norm(cluster_samples-center,dim=1)
#             var =  torch.mean(torch.clamp(s1,1,50))
#             dists = dists + var
#         loss = dists
#         return loss
    
class PEMLoss(nn.Module):
# PTM loss funtion
    def __init__(self,g1,b0=0.3):
        super(PEMLoss, self).__init__()
        self.g1 = g1
        self.b0 = b0
    def forward(self,x,y):
        d = F.pdist(x)
        y=torch.unsqueeze(y, 1)
        gamma = F.pdist(y)
        N = gamma.shape[0]
        gamma = torch.where(gamma==0,torch.ones(N),self.g1*torch.ones(N))
        loss = torch.mean(pt(gamma*d+self.b0))
        return loss

    
def train(model,optimizer,data,labels,g1=0.1):
    model.train()
    data = torch.tensor(data,dtype = torch.float32)
    y = torch.tensor(labels,dtype = torch.float32)
    pem_fcn = PEMLoss(g1)
    for e in range(501):
        dx = model(data)
        loss = pem_fcn(dx,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e%100==0:
            print('loss at ',e,' is',loss.item())
    model.eval()
    return loss.item()

class Data():
    def __init__(self):
        pass
    
    def read_data(self,base_features_path,input_dim,start,ender,types='mini'):
        if types == 'mini':
            features,labels,means,covs = self.mini(base_features_path,input_dim,start,ender)
        if types == 'cub':
            features,labels,means,covs = self.cub(base_features_path,input_dim,start,ender)
        return features,labels,means,covs
            
    def mini(self,base_features_path,input_dim,start,ender):
        base_means = []
        base_cov = []
        idx = 0
        label_idx = start
        f_matrix = np.empty((0,2048))  # Copy operation
        labels = np.empty((0))
        with open(base_features_path, 'rb') as f:
            data = pickle.load(f)
            for key in data.keys():
                if label_idx==ender:
                    break
                feature = np.array(data[key])
                label = (label_idx)*np.ones((feature.shape[0],1))
                idx = idx + feature.shape[0]
                feature = np.power(feature,1)
                mean = np.mean(feature, axis=0)
                cov = np.cov(feature.T)
                base_means.append(mean)
                base_cov.append(cov)
                f_matrix= np.vstack((f_matrix,feature))
                label = label_idx * np.ones(feature.shape[0])
                labels = np.hstack((labels,label))
                print(idx,label_idx)
                label_idx = label_idx+1
        labels = labels.squeeze()
        base_means = np.array(base_means)
        base_cov = np.array(base_cov)
        return f_matrix,labels,base_means,base_cov
    
    def cub(self,base_features_path,input_dim,start,ender):
        base_means = []
        base_cov = []
        idx = 0
        label_idx = start
        f_matrix = np.empty((0,640))  # Copy operation
        labels = np.empty((0))
        with open(base_features_path, 'rb') as f:
            data = pickle.load(f)
            for key in data.keys():
                if label_idx==ender:
                    break
                feature = np.array(data[key])
                label = (label_idx)*np.ones((feature.shape[0],1))
                idx = idx + feature.shape[0]
                feature = np.power(feature,0.8)
                mean = np.mean(feature, axis=0)
                cov = np.cov(feature.T)
                base_means.append(mean)
                base_cov.append(cov)
                f_matrix= np.vstack((f_matrix,feature))
                label = label_idx * np.ones(feature.shape[0])
                labels = np.hstack((labels,label))
                print(idx,label_idx)
                label_idx = label_idx+1
        labels = labels.squeeze()
        base_means = np.array(base_means)
        base_cov = np.array(base_cov)
        return f_matrix,labels,base_means,base_cov
    
    def data_generator(self,classes,x,labels,shot_num):
        ns = x.shape[1]
        samples = np.empty((0,ns))
        sam_labels = np.empty((0))
        sam_means = np.empty((0,ns))
        for j,i in enumerate(classes):
            mask = labels == int(i)
            cluster = x[mask]
            idx=np.random.permutation(cluster.shape[0])[0:shot_num]
            sample = cluster[idx]
            samples = np.vstack((samples,sample))
            sam_labels = np.hstack((sam_labels,j*np.ones((shot_num))))
        sam_labels = sam_labels.squeeze()
        return samples,sam_labels
    
class Distill():
    def __init__(self,class_num):
        self.class_num = class_num
        
    def sigma_points(self,x):
        n = x.shape[1]
        data = np.zeros((2*n+1,n))
        mu = np.mean(x, axis=0)
        cluster = x - mu
        sigma = np.cov(cluster.T)
        sigma = sigma*sigma
        data[0] = mu
        lamb = 1*(n+20)-n
        for i in range(1,n):
            data[i]=mu+np.sqrt((n+lamb)*sigma)[i,:]
            data[n+1+i]=mu-np.sqrt((n+lamb)*sigma)[i,:]
        return data
        
    def select_sample(self,x,y,num =20):
        index = []
        label = []
        for i in range(self.class_num):
            mask = y == i
            cluster = x[mask]
            sigmas = self.sigma_points(cluster)
#             print(sigmas)
            m = sigmas.shape[0]
            n = cluster.shape[0]
            matrix = np.zeros((m,n))
            for mm in range(m):
                for nn in range(n):
                    matrix[mm,nn] = np.sqrt(np.sum(sigmas[mm]-cluster[nn])**2)
            for _ in range(num):
                try:
                    idx = np.argmin(matrix)
                    r = idx//n
                    c = idx%n
#                     print(r,c,i)
                    matrix[r,:] = 1e8
                    matrix[:,c] = 1e8
                except:
                    pass
                index.append(c)
                label.append(i)
        return np.array(index),np.array(label)
    
    def refine(self,x,y,idx,idy):
        data = np.empty((0,x.shape[1]))
        for i in range(self.class_num):
            mask = y == i
            cluster = x[mask]
            mask = idy == i
            c_idx = idx[mask]
            ds = cluster[c_dix]
            data = np.np.vstack((data,ds))
        return data
        
    def upsampling(self,x,y,num,bias=1):
        data = np.zeros((self.class_num*num,x.shape[1]))
        label = np.empty((0))
        for i in range(self.class_num):
            mask = y == i
            cluster = x[mask]
            mu = np.mean(cluster, axis=0)
            cluster = cluster - mu
            sigma = np.cov(cluster.T)+bias
#             sigma = torch.norm(cluster-mu,dim=1)+0.2
            samples = np.random.multivariate_normal(mu, sigma, num)
            data[i*num:(i+1)*num] = samples
            label = np.hstack((label,i*np.ones((num))))
        label = label.squeeze()
        print(data.shape,label.shape)
        return data,label
    
    def upsampling_few(self,x,y,num,bias=0.5):
        data = np.zeros((5*num,x.shape[1]))
        label = np.empty((0))
        for i in range(5):
            mask = y == i
            cluster = x[mask]
            mu = np.mean(cluster, axis=0)
#             cluster = cluster
            sigma = np.cov(cluster.T)+bias*np.eye(x.shape[1])
            samples = np.random.multivariate_normal(mu, sigma, num)
            data[i*num:(i+1)*num] = samples
            label = np.hstack((label,i*np.ones((num))))
        label = label.squeeze()
        print(data.shape,label.shape)
        return data,label
    
    def upsampling_full(self,x,y,means,covs,num,bias=1):
        data = np.zeros((self.class_num*num,x.shape[1]))
        label = np.empty((0))
        for i in range(self.class_num):
            mask = y == i
            cluster = x[mask]
            mu = means[i]
            sigma = covs[i]+bias*np.eye(x.shape[1])
            samples = np.random.multivariate_normal(mu, sigma, num)
            data[i*num:(i+1)*num] = samples
            label = np.hstack((label,i*np.ones((num))))
        label = label.squeeze()
        print(data.shape,label.shape)
        return data,label
    
class FewShoot():
    def __init__(self,class_num):
        self.upper = Distill(class_num)
    
    def booster_sample(self,few_x,few_y,tx,ty,params,ways = 5):
        upper = self.upper
        INPUT_DIM = few_x.shape[1]
        x_up,y_up = upper.upsampling_few(few_x,few_y,params['sam_num'],bias = 0.00)
        model = MCC(input_dim=INPUT_DIM, output_dim=params['output_dim'])
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['decay'])
        train(model,optimizer,x_up,y_up,params['g1'])
        W1 = model.fc1.weight.clone().detach().numpy()
        x_r_new = x_up@W1.T
        x_t_new = tx@W1.T
        acc = logistic_test(x_r_new,y_up,x_t_new,ty)
#         print("The acc of PEM (logistic) is:",acc)
        return acc


    def booster_multi(self,f_x,f_y,t_x,t_y,params,ways = 5):
        ress = np.empty((times,t_x.shape[0]))
        INPUT_DIM = few_x.shape[1]
        for i in range(params['times']):
            model = MCC(input_dim=INPUT_DIM, output_dim=params['output_dim'])
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['decay'])
            train(model,optimizer,x_up,y_up,params['g1'])
            W1 = model.fc1.weight.clone().detach().numpy()
            x_r_new = f_x@W1.T
            x_t_new = t_x@W1.T
            classifier = LogisticRegression(max_iter=1000).fit(X=x_r_new, y=f_y)
            res = classifier.predict(x_t_new)
            ress[i] = res
        res = stats.mode(ress, axis = 0)[0]
        idx = np.where(res == t_y)[0]
        acc = idx.shape[0]/t_y.shape[0] 
#         print("The acc of PEM (logistic) is:",acc)
        return acc

