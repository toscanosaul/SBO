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
   # mu = [1.72004110e+00, 1.01494883e+00, 5.52864499e+02, 9.34277539e+02]
    num_dims = 5

   # XWtrain = np.loadtxt("training_data/XWtrain_100_points.txt")
    #yTrain = np.loadtxt("training_data/yTrain_100.txt")
    #yTrain = yTrain.reshape((XWtrain.shape[0], 1))
    #noise = np.loadtxt()

    yTrain = np.loadtxt("observed_values1.txt")
    XWtrain = np.loadtxt("observed_inputs1.txt")
    noise = np.loadtxt("noise1.txt")
    noise = (noise**2)/30.0
    noise = np.zeros(len(noise))
   # XWtrain = XWtrain.astype(float)
    new_point = XWtrain[99:100,:]
    XWtrain = XWtrain[0:99, :]
    yTrain = yTrain[0:99]
   # noise = noise[0:99]
    #print XWtrain
    #XWtrain = XWtrain[0:11,:]
    #yTrain = yTrain[0:11]
    #noise = noise[0:11]
    noise = 30.339
    noise = 0.0
    data = {}
    data['X_data'] = XWtrain
    data['y'] = yTrain
    data['noise'] = noise
  #  data['noise'] = noise
    data['noiseless'] = True  #noiseless if true we get the parameters using mle
    data['matern'] = mu
    data['log_multiKernel'] = log_multi
    data['nEvals'] = 1
    data['dim_x'] = 4
    data['dim_w'] = 1
    data['n_restarts_an'] = 10
    data['evaluation_f'] = g

    lower = [0.1, 0.01, 1, 1, 0]
    upper = [51, 1.01, 21, 201, 5]

    nGrid = [7, 6, 11, 6]

    domainX = []
    for i in range(4):
      domainX.append(np.linspace(lower[i], upper[i], nGrid[i]))

    points = [[a, b, c, d] for a in domainX[0] for b in domainX[1] for c in
              domainX[2] for d in domainX[3]]

    points = np.array(points)


    domain = []
    for i in range(5):
      dict={}
      dict['lower'] = lower[i]
      dict['upper'] = upper[i]
      domain.append(dict)

    data['domain'] = domain
    data['type_domain'] = ['real', 'real', 'integer', 'integer', 'integer']
    data['candidate_points'] = points

  #  noise = 2.0
    model = K_Folds(num_dims, 5, **data)
    model.run_sbo()
    dfg
    self=model

    noise_an = self.noise * np.ones(self.observed_inputs.shape[0])

    self.get_optimal_point()

    self.SBO_stats.setup(
      XW=self.observed_inputs,
      noise=noise_an
    )
    print new_point[0:1,0:4]


#x, Xhist, yHist, gradient = True, onlyGradient = False,

    z,grad =model.SBO_stats.aN_grad(
      new_point[0:1, 0:4],
      self.observed_inputs,
      self.observed_values,
      gradient=True
    )

    dh = 0.00000001
    new_point_2 = np.copy(new_point[0:1, 0:4])
    # print new_point_2
    new_point_2[0, 3] += dh

    z_= model.SBO_stats.aN_grad(
      new_point_2,
      self.observed_inputs,
      self.observed_values,
      gradient=False)

    print "Grad"
    print (z_ -z)/dh

    print grad



    #print model.observed_inputs.shape
   # model.run_sbo(1)

    ddf
    #model.mle_parameters(n_restarts=50)

    sbo = model.VOI

    self=model
    noise_voi = self.noise * np.ones(self.observed_inputs.shape[0])
    sbo.setup(
          XW=self.observed_inputs,
          y=self.observed_values,
          noise=noise_voi
    )


   # print model.mle_parameters()
 #   model.noise = np.exp(noise)
 #   model_2 = K_Folds(num_dims, 5, **data)
 #   print model.log_likelihood()
 #   print model.grad_log_likelihood()
 #   dh = 0.0001
 #   model_2.noise += np.exp(noise+dh)
 #   print model_2.log_likelihood()
 #   print (model_2.log_likelihood() - model.log_likelihood())/dh


#    print model.log_likelihood()


    #model.get_training_data(900, signature='2')


    z= model.VOI.VOIfunc(
      pointNew=new_point,
      grad=True,
      XW=XWtrain
    )
    print "this gradient is"
    print z

    dhs=[10e16,10e14,10e13,10e12,10e11,10e10,10e9,10e8,10e7,10e6,10e5,10e4,1000.0,1.0,0.1,0.01,0.001, 0.0001, 1e-5,1e-6,1e-7,1e-8,1e-9,1e-10, 1e-11, 1e-12, 1e-13, 1e-15]

    for dh in dhs:
      new_point_2 = np.copy(new_point)
     # print new_point_2
      new_point_2[0,2] += dh
     # print new_point_2


      z_ = model.VOI.VOIfunc(
        pointNew=new_point_2,
        grad=False,
        XW=XWtrain
      )
     # print z_
      print "grad"
      print (z_-z[0])/dh
      ####Check gradients. Why voi is zero in 0,2
    #print model.evaluate_function([10,    1.0,    10        ,  166.        ,    1        ])
   # model.get_training_data(100)

    # model.mle_parameters(n_restarts=2)


  #  r= model.cross_validation_mle_parameters(
  #      XWtrain,
  #      yTrain,
  #      noise,
  #      n_restarts=30
  #    )

   # np.savetxt("means_diag.txt",r[2])
   # np.savetxt("std_diag.txt",r[3])


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

