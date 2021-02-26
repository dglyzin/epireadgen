import torch
import torch.distributions as dist
import matplotlib.pyplot as plt


def test_bayes_EM():
    '''solving 20.3.2. Russel-Norving'''
    f = lambda count, h1, h2, h3, b1: count*h1*h2*h3*0.6/(h1*h2*h3*0.6 +(1-h1)*(1-h2)*(1-h3)*0.4)*0.001 if b1 else count*(1-h1)*(1-h2)*(1-h3)*0.4/(h1*h2*h3*0.6+(1-h1)*(1-h2)*(1-h3)*0.4)*0.001   
    result = sum([f(count, h1, h2, h3, 1)
                  for count, h1, h2, h3 in [(273, 0.6, 0.6, 0.6),
                                            (79, 0.4, 0.6, 0.6),
                                            (93, 0.4, 0.6, 0.6),
                                            (100, 0.4,0.4,0.6),
                                            (104, 0.4, 0.6, 0.6),
                                            (94, 0.4, 0.4, 0.6),
                                            (90,0.4, 0.4, 0.6),
                                            (167, 0.4, 0.4, 0.4)]])
    return(result)


def learn_cs_em(observed, steps, imu, isigma, iw, cs_counts=3,):
    
    N = len(observed)

    # dublicate observed for each class:
    observed = torch.cat([torch.unsqueeze(observed, 0)
                          for k in range(cs_counts)], 0).T
    observed_cs = torch.cat(
        [torch.unsqueeze(torch.arange(0, cs_counts), 0)
         for k in range(N)], 0)
    # print("observed:")
    # print(observed.shape)
    # print("observed_cs:")
    # print(observed_cs.shape)

    # init values
    
    mu = imu
    # mu = torch.tensor([1., 3.5, 4.0])
    sigma = isigma
    # sigma = torch.tensor([1., 1.0, 1.0])
    w = iw
    # w = torch.tensor([0.1, 0.2, 0.7])
    gen_p_xj_ci = lambda mu, sigma: dist.Normal(mu, sigma)
    gen_p_ci = lambda w: dist.Binomial(cs_counts, w)

    mus, sigmas, wis = [], [], []

    for step in range(steps):
        dist_xj_ci = gen_p_xj_ci(mu, sigma)
        dist_ci = gen_p_ci(w)
        
        # FOR computing p_ci_xj:
        # print("test1")
        # print(dist_ci.sample())
        # print(dist_xj_ci.log_prob(observed).shape)
        # print(dist_ci.log_prob(observed_cs.float()).shape)
        
        mult = torch.exp(dist_xj_ci.log_prob(observed)
                         + dist_ci.log_prob(observed_cs.float()))
        
        # normalization constant:
        # print(mult)
        
        alpha = mult.sum(1)**(-1.)
        # print("alpha")
        # print(alpha)
        
        p_ci_xj = (alpha * mult.T).T
        # print("p_ci_xj:")
        # print(p_ci_xj.shape)
        # print(p_ci_xj)
        
        # END FOR

        ni = p_ci_xj.sum(0)
        # print("ni:")
        # print(list(ni.shape))
        # print(ni)
        
        mu = (p_ci_xj * observed).sum(0)/ni
        # print("mu:")
        # print(list(mu.shape))
        # print(mu)

        sigma = (p_ci_xj * (observed-mu)**2).sum(0)/ni
        # print("sigma:")
        # print(sigma.shape)
        # print(sigma)

        wi = ni/N
        # print("wi:")
        # print(wi.shape)
        # print(wi)
        # return
        mus.append(mu)
        sigmas.append(sigma)
        wis.append(wi)
    # plt.scatter(a,b)
    
    return(torch.cat([torch.unsqueeze(mu, 0)
                      for mu in mus], 0),
           torch.cat([torch.unsqueeze(sigma, 0)
                      for sigma in sigmas], 0),
           torch.cat([torch.unsqueeze(wi, 0)
                      for wi in wis], 0))


def test_learn_cs_em():
    steps = 4

    imu = torch.tensor([1., 2., 3.])
    isigma = torch.tensor([0.3, 0.3, 1.])
    iwis = torch.tensor([0.1, 0.4, 0.5])
    observed = sample_cs(100, imu, isigma, iwis)

    limu = torch.tensor([1., 3.5, 4.0])
    lisigma = torch.tensor([1., 1.0, 1.0])
    liwi = torch.tensor([0.1, 0.2, 0.7])
    mus, sigmas, wis = learn_cs_em(observed, steps,
                                   limu, lisigma, liwi)

    print("initial model:")
    print((imu, isigma, iwis))

    print("lerned initial model:")
    print((limu, lisigma, liwi))
    
    print("learned model:")
    print((mus[-1], sigmas[-1], wis[-1]))
    
    # FOR plot:
    fig, axs = plt.subplots(3, 3)
    names = ["mu", "sigma", "wi"]
    data = [mus, sigmas, wis]

    for i in range(3):
        name = names[i]
        for j in range(3):
            axs.flat[3*i+j].plot(data[i].T[j])
            axs.flat[3*i+j].set_title("%s %d" % (name, j))
    plt.show()
    # END FOR


def sample_cs(steps, mu, sigma, cs_probs):
    dist_ci = dist.Categorical(cs_probs)
    dist_xj_ci = dist.Normal(mu, sigma)
    cs_sampler = lambda: dist_xj_ci.sample()[dist_ci.sample()]
    return(torch.tensor([cs_sampler() for i in range(steps)]))


def test_sample_cs(mu, sigma, cs_probs):
    import matplotlib.pyplot as plt
    
    result = sample_cs(1000, mu, sigma, cs_probs)

    plt.hist(result, 70)
    plt.show()


if __name__ == "__main__":

    test_learn_cs_em()
    # test_sample_cs(torch.tensor([0.0, 0.5, 1.0]),
    #                torch.tensor([0.1, 0.1, 0.7]),
    #                torch.tensor([0.2, 0.2, 0.6]))
