import numpy as np
from models import K_Folds
from pmf import cross_validation,PMF


if __name__ == '__main__':
  #  XWtrain = np.loadtxt("training_data/XWdata.txt")
  #  yTrain = np.loadtxt("data/ydata.txt").reshape((XWtrain.shape[0], 1))
  #  XWtrain_2 = np.loadtxt("data/XWdata_2.txt")
  #  yTrain2 = np.loadtxt("data/ydata_2.txt").reshape((XWtrain_2.shape[0],1))
  #  XWtrain = np.concatenate((XWtrain,XWtrain_2))
  #  yTrain = np.concatenate((yTrain,yTrain2))

    train =[]
    test = []
    validate=[]
    data_all =[]
    for i in range(1, 6):
      data = np.loadtxt("data/u%d.base" % i)
      test = np.loadtxt("data/u%d.test" % i)
      train.append(data)
      validate.append(test)
      data_all.append(np.concatenate((data, test), axis=0))
    num_user = 943
    num_item = 1682

    def g(x, w, random_seed=1):
      np.random.seed(random_seed)
      indexes = np.arange(5)
      w=int(w[0])
      indexes = np.delete(indexes, w)
      train_data = data_all[indexes[0]]
      indexes = np.delete(indexes, indexes[0])
      for i in indexes:
        train_data = np.concatenate((train_data, data_all[i]), axis=0)
      val = PMF(num_user, num_item, train_data, data_all[w], x[0], x[1], int(x[3]),
                int(x[2]))
      return -val * 100.0

    log_mu = np.loadtxt("data/mean_log_mu.txt")
    log_multi = np.loadtxt("data/mean_log_cov_folds.txt")


    log_mu = np.array([-1.43285632, -1.04592418, -27.03837878, -6.839643])

    log_multi = np.array([5.09945529e+00,   3.14705582e-01 ,  2.76955350e+00 ,  2.59006831e+00,
           2.72541135e+00  ,-1.59429134e+01  ,-3.89236520e+00  , 2.81947210e+00,
          -1.41557863e+01 , -2.00693301e+01 ,  3.85411365e+00  , 2.46330959e+00,
          -1.79366753e+01 , -1.65765782e+01 , -1.79260710e+01])

    mu = [1.72004110e+00, 1.01494883e+00 ,  5.52864499e+11 ,  9.34277539e+02]
    num_dims = 5

   # XWtrain = np.loadtxt("training_data/XWtrain_100_points.txt")
    #yTrain = np.loadtxt("training_data/yTrain_100.txt")
    #yTrain = yTrain.reshape((XWtrain.shape[0], 1))
    #noise = np.loadtxt()

    yTrain = np.loadtxt("observed_values1.txt")
    XWtrain = np.loadtxt("observed_inputs1.txt")
    noise = np.loadtxt("noise1.txt")
    noise = (noise**2)/30.0
    #print XWtrain
    XWtrain = XWtrain[0:11,:]
    yTrain = yTrain[0:11]
    noise = noise[0:11]
    data = {}
    data['X_data'] = XWtrain
    data['y'] = yTrain
    data['noise'] = noise
    data['matern'] = mu
    data['log_multiKernel'] = log_multi
    data['nEvals'] = 1
    data['dim_x'] = 4
    data['dim_w'] = 1
    data['evaluation_f'] = g

    lower = [0.1, 0.01, 1, 1, 0]
    upper = [51, 1.01, 21, 201, 5]

    domain = []
    for i in range(5):
      dict={}
      dict['lower'] = lower[i]
      dict['upper'] = upper[i]
      domain.append(dict)

    data['domain'] = domain
    data['type_domain'] = ['real', 'real', 'integer', 'integer', 'integer']
    model = K_Folds(num_dims, **data)

    #print model.evaluate_function([10,    1.0,    10        ,  166.        ,    1        ])
   # model.get_training_data(100)


    print model.cross_validation_mle_parameters(
      XWtrain,
      yTrain,
      noise
    )

   # print (f2-f1)/dh - model._kernel.gradient(XWtrain)[17]
  #  print "grad"
 #   print model._kernel.gradient(XWtrain)[0]
#    print model.log_likelihood()
  #  print (model_4.log_likelihood() - model.log_likelihood())/dh
   # print model.grad_log_likelihood()

  #  print model._kernel.ls.value
   # print "opt"
   # print model.mle_parameters(n_restarts=2)
    #print "param"
    #print model._kernel.ls.value

    #print model._collect_samples(1)

