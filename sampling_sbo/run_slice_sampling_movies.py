import numpy as np
from models import K_Folds


if __name__ == '__main__':
    XWtrain = np.loadtxt("data/XWdata.txt")
    yTrain = np.loadtxt("data/ydata.txt").reshape((XWtrain.shape[0], 1))
    XWtrain_2 = np.loadtxt("data/XWdata_2.txt")
    yTrain2 = np.loadtxt("data/ydata_2.txt").reshape((XWtrain_2.shape[0],1))
    XWtrain = np.concatenate((XWtrain,XWtrain_2))
    yTrain = np.concatenate((yTrain,yTrain2))

    log_mu = np.loadtxt("data/mean_log_mu.txt")
    log_multi = np.loadtxt("data/mean_log_cov_folds.txt")

    log_mu = np.array([-1.43285632, -1.04592418, -27.03837878, -6.839643])

    log_multi = np.array([5.55312055, 2.83900732, 3.800882, 4.06687021,
                         3.73810855, -15.94291345, 2.95538257, 3.60834691,
                         -14.15578629, -20.06933011, 4.76588958, 3.42354678,
                         -17.9366753, -16.57657817, -17.92607099])
    num_dims = 5

    data = {}
    data['X_data'] = XWtrain
    data['y'] = yTrain[:,0]
    data['noise'] = np.zeros(len(yTrain[:,0]))
    data['matern'] = np.exp(-log_mu)
    data['log_multiKernel'] = log_multi

    model = K_Folds(num_dims, **data)

    model_4 = K_Folds(num_dims, **data)


#    model._burn_samples(100)
    np.random.seed(100)

    dh =0.000001
    log_multi_2 = np.copy(log_multi)
    log_multi_2[13] += dh
    data['log_multiKernel'] = log_multi_2
    model_2 = K_Folds(num_dims, **data)


    mu_2 = np.copy(np.exp(-log_mu))
    mu_2[0] += dh
    data['log_multiKernel'] = log_multi
    data['matern'] = mu_2
    model_3 = K_Folds(num_dims, **data)

    model_4.mean.value = model.mean.value + dh

    f2 = model_2.log_likelihood()
    f1 = model.log_likelihood()

    f1 = model._kernel.cov(XWtrain)
    f2 = model_2._kernel.cov(XWtrain)


   # print (f2-f1)/dh - model._kernel.gradient(XWtrain)[17]
    print "grad"
 #   print model._kernel.gradient(XWtrain)[0]
#    print model.log_likelihood()
  #  print (model_4.log_likelihood() - model.log_likelihood())/dh
   # print model.grad_log_likelihood()

    print model._kernel.ls.value
    print "opt"
    print model.mle_parameters(n_restarts=60)
    print "param"
    print model._kernel.ls.value
