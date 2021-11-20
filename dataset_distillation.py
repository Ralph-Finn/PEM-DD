# The python files for dataset distillation
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import imp
import utils
import torch.optim as optim
import numpy as np
# imp.reload(utils)
from sklearn.manifold import TSNE

# ArgParse
parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--g1', type=float, default=0.1)
parser.add_argument('--data_path', default='./data/data.plk', type=str)
parser.add_argument('--output_dim', type=int, default=10)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--class_num', type=int, default=10)
parser.add_argument('--split', type=int, default=25)
parser.add_argument('--dataset_type',  type=str, default='mini')
parser.add_argument('--seed', type=int, default=123)


args = parser.parse_args()
if args.dataset_type == 'cub':
    INPUT_DIM = 640
if args.dataset_type == 'mini':
    INPUT_DIM = 2048
   
OUTPUT_DIM = args.output_dim
#base_features_path = "./data/novel_features.plk"
base_features_path = args.data_path

datater = utils.Data()
f_matrix,labels,means,covs = datater.read_data(base_features_path,INPUT_DIM,0,args.class_num,args.dataset_type)
f_matrix = (f_matrix-np.min(f_matrix))/np.max(f_matrix)
print(means.shape)
x_train,x_test,y_train,y_test = utils.split_dataset(f_matrix,labels,4) 
print(x_train.shape,x_test.shape)
_,x_some,_,y_some = utils.split_dataset(x_train,y_train,args.split)
print(x_some.shape)


def main():
    acc = utils.logistic_test(x_train,y_train,x_test,y_test)
    acc_some = utils.logistic_test(x_some,y_some,x_test,y_test)
    print("Logistic Acc (all):{0},Logistic Acc (some):{1}".format(acc,acc_some))
    
    #################
    model = utils.MCC(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    utils.train(model,optimizer,x_train,y_train,args.g1)
    W1 = model.fc1.weight.clone().detach().numpy()
    
    #######
    
#     f_new = f_matrix@W1.T
#     tsne = TSNE()
#     out = tsne.fit_transform(f_new)
#     plt.figure(figsize=(10,10))
#     for i in range(args.class_num):
#         indices = labels == i
#         x, y = out[indices].T
#         plt.scatter(x, y, label=str(i))
#     plt.legend()
#     plt.show()
    
    #########
    x_r_new = x_some@W1.T
    x_t_new = x_test@W1.T
    acc = utils.logistic_test(x_r_new,y_some,x_t_new,y_test)
    print("The acc of PEM (logistic) is:",acc)
    
    ##########
    upper = utils.Distill(args.class_num)
    x_id,y_id = upper.select_sample(x_train@W1.T,y_train)
    print(x_id,y_id)


if __name__ == '__main__':
    main()
