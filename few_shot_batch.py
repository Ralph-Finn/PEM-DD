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
parser.add_argument('--g1', type=float, default=0.5)
parser.add_argument('--data_path', default='./data/data.pkl', type=str)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--class_num', type=int, default=10)
parser.add_argument('--split', type=int, default=2)
parser.add_argument('--sample_num', type=int, default=200)
parser.add_argument('--output_dim', type=int, default=10)
parser.add_argument('--mul_num', type=int, default=10)
parser.add_argument('--test_num', type=int, default=100)
parser.add_argument('--boost_type', type=str, default="sample")
parser.add_argument('--dataset_type',  type=str, default='mini')
parser.add_argument('--seed', type=int, default=123)


args = parser.parse_args()
if args.dataset_type == 'cub':
    INPUT_DIM = 640
if args.dataset_type == 'mini':
    INPUT_DIM = 2048
    
OUTPUT_DIM = args.output_dim
base_features_path = args.data_path

datater = utils.Data()
f_matrix,labels,means,covs = datater.read_data(base_features_path,INPUT_DIM,0,args.class_num,args.dataset_type)
f_matrix = (f_matrix-np.min(f_matrix))/np.max(f_matrix)
print(means.shape)
x_train,x_test,y_train,y_test = utils.split_dataset(f_matrix,labels,2) 
print(x_train.shape,x_test.shape)

FT = utils.FewShoot(args.class_num)
bt = args.boost_type

# One time test function
def one_test(params,test_num):
    accs = []
    for i in range(test_num):
        random_index=np.random.permutation(np.arange(args.class_num))[0:5]
        f_x,f_y = datater.data_generator(random_index,x_train,y_train, 5)
        t_x,t_y = datater.data_generator(random_index,x_test,y_test,50)
        if bt =='sample': 
            n_acc = FT.booster_sample(f_x,f_y,t_x,t_y,params,ways=5)
        if bt =='multi':
            n_acc = FT.booster_multi(f_x,f_y,t_x,t_y,params,ways=5)
        print("ACC:",n_acc)
        accs.append(n_acc)
    accs = np.array(accs)
    return np.mean(accs)

def main():
    params = {'learning_rate': args.lr,
     'output_dim': OUTPUT_DIM,
     'decay': args.decay,
     'g1': args.g1,
     'sam_num': args.sample_num,
      'times':args.mul_num}
    
    acc_total = one_test(params,args.test_num)
    print("The avarange accuracy is",acc_total)


if __name__ == '__main__':
    main()
