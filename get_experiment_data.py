import numpy as np
import argparse

np.random.seed(20)


"""
Efficient Variational Inference for Sparse Deep
Learning with Theoretical Guarantee
data_size = 3000, data_dim = 200
"""
def Bai_data(data_size, data_dim, sigma_noise):
    X = np.random.uniform(-1, 1, (data_size, data_dim))
    epsilon_y = np.random.normal(0, sigma_noise, X.shape[0])
    y = 7 * X[:, 1] / (1 + X[:, 0]**2) + 5 * np.sin(X[:, 2] * X[:, 3]) + 2 * X[:, 4] + epsilon_y
    gamma = np.zeros(data_dim)
    gamma[:5] = 1
    return X, y, gamma


"""
Bayesian Neural Networks for Selection of Drug Sensitive Genesから
data_size = 200, data_dim = 500
"""
def Liang_data(data_size, data_dim, sigma_noise):
    # covariance_matrix = np.where(np.ones(data_dim)==1, 1, 0.5)
    
    e = np.random.randn(1)
    Z = np.random.multivariate_normal(np.zeros(data_dim), np.ones((data_dim, data_dim)), data_size)
    X = (e + Z)/2
    epsilon_y = np.random.normal(0, sigma_noise, X.shape[0])
    y = (10 * X[:, 1])/(1 + X[:, 0]**2) + 5 * np.sin(X[:, 2] * X[:, 3]) + 2 * X[:, 4] + epsilon_y
    gamma = np.zeros(data_dim)
    gamma[:5] = 1
    return X, y, gamma


def main(args):
    data_size, data_dim = args.data_size, args.data_dim
    sigma_noise = 1.0
    X_list = []
    y_list = []
    for _ in range(args.data_num):
        if args.which_data == 'Bai':
            X, y, gamma = Bai_data(data_size, data_dim, sigma_noise)
        elif args.which_data == 'Liang':
            X, y, gamma = Liang_data(data_size, data_dim, sigma_noise)
        else:
            assert False
        
        X_list.append(X)
        y_list.append(y)

    filename = f"s{args.data_size}_d{args.data_dim}_n{args.data_num}_{args.which_data}"
    np.savez(f'./' + filename + '.npz', X_list, y_list, gamma)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument('--data_size', type=int, default=1000)
    parser.add_argument('--data_dim', type=int, default=20)
    parser.add_argument('--which_data', type=str, choices=['Bai', 'Liang'])
    parser.add_argument('--data_num', type=int, default=1)
    args = parser.parse_args()
    main(args)
    print('Done!')


