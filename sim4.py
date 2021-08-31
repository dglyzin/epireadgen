import torch
import pyro
import pyro.distributions as pdist
from pyro.ops.indexing import Vindex
import sim3 as sm3


# FOR model1:
def model1(obs=torch.tensor([1., 1.])):
    '''
    Model:
    {a| a=Uniform(0, 1)}
      ->{Bag| [p(Bug=0)=1-a, p(Bug=1)=a]}
       ->{Color| [p(Color|"Bag": 0)=[0.1, 0.9],
                  p(Color|"Bag": 1) = [0.5, 0.5]]}
    Ask what p(X|cond) will be for X is some of [a, Bag, Color]
    
    # accurate P(B|a>=0.7) =P(B, a>=0.7)/p(a>=0.7) =  <0.15, 0.85>
    '''

    # +1 is reserve for t==0:
    T = len(obs) + 1

    # dist is fake, only obs important:
    pyro.sample("T", pdist.Uniform(0, 1),
                obs=torch.tensor(T))

    zdinit = torch.tensor(0.5)
    transition = torch.tensor([0.3, 0.7])
    emission = torch.tensor([0.2, 0.9])
    '''
    pt0 = pyro.sample("pt0", pdist.Uniform(0, 1))
    pt1 = pyro.sample("pt1", pdist.Uniform(0, 1))
    pe0 = pyro.sample("pe0", pdist.Uniform(0, 1))
    pe1 = pyro.sample("pe1", pdist.Uniform(0, 1))

    transition = torch.tensor([pt0, pt1])
    emission = torch.tensor([pe0, pe1])
    '''
    for t in range(T):
        # j = K - i % K
        if t == 0:
            z_prev = pyro.sample("z_0", pdist.Bernoulli(zdinit))
            continue

        ip_transition = transition[z_prev.type(torch.long)]
        z = pyro.sample("z_%d" % t,
                        pdist.Bernoulli(ip_transition))
        z_prev = z
        
        ip_emission = emission[z.type(torch.long)]
        dist_x = pdist.Bernoulli(ip_emission)
        x = pyro.sample("x_%d" % t,
                        dist_x,)  # obs=xs_batch[:, t])

        x_obs = pyro.sample("ox_%d" % t,
                            pdist.Bernoulli(ip_emission),
                            obs=obs[t-1])
    return(x)


def m1_fcond(trace):
    '''check if x[t] == obs[t] for all t'''
    T = trace.nodes["T"]["value"]
    # print(trace.nodes)
    for t in range(T)[1:]:
        x = trace.nodes["x_%d" % t]["value"]
        # print("x:", x)
        x_obs = trace.nodes["ox_%d" % t]["value"]
        # print("x_obs:", x_obs)
        if not (x == x_obs).all():
            return(False)
    return(True)


def run_m1(model=model1, fcondition=m1_fcond, N=3,
           obs=[1.], delta=100, plot_zs=True):

    '''
    Model:
    {zt|}
      ->{zt1|Bernoulli(transition[zt])}
      ->{xt(=obs)|Bernoulli(emission[zt])}

    Plot only convergence of z (x is a data and hence fixed)
    -- ``N`` - count of steps
    -- ``delta`` - samples in each step'''

    T = len(obs)+1
    xs = [torch.tensor([])]*T
    zs = [torch.tensor([])]*T
    
    res0 = [[] for t in range(T)]
    # this is a mistake (same inner array for all t):
    # res1 = [[]]*T

    steps = int(N/delta)
    for step in range(steps):
    
        print("step %d from %d " % (step, steps))
        samples = sm3.rejection_sampling1(
            model, {"obs": torch.tensor(obs)}, fcondition, delta)
        n = len(samples)
        print("len(samples): ", n)

        if n > 0:
            for strace in samples:
                for t in range(T):
                    if t != 0:
                        xs[t] = torch.cat([xs[t],
                                           strace.nodes["x_%d" % t]['value'].unsqueeze(0)])
                    zs[t] = torch.cat([zs[t],
                                       strace.nodes["z_%d" % t]['value'].unsqueeze(0)])

            n = len(zs[-1])
            print("\nfull len(samples): ", n)
            print("zs:")
            for t in range(T):
                rz = (len(zs[t][zs[t] == 0.])/n, len(zs[t][zs[t] != 0.])/n)
                print("z_%d: " % t, rz)
                res0[t].append(rz[0])
                # res1[t].append(rz[1])
                
            print("\nxs:")
            for t in range(T):
                if t != 0:
                    rx = (len(xs[t][xs[t] == 0.])/n, len(xs[t][xs[t] != 0.])/n)
                    print("x_%d: " % t, rx)

    if plot_zs:
        import matplotlib.pyplot as plt
        for t in range(T):
            # if t > 0:
            print("plot z_%d" % t)
            # print(res0[t])
            plt.plot(res0[t])
            # plt.plot(res1[t])
            plt.show()
    # df = sm3.make_dataFrame(samples)
    # sm3.plot_results(df)
    # return(df)
# END FOR


def hmm_target(z0, T_seq, T_bseq, K):
    '''
    - ``t_seq`` -- general sequence length
    i.e. global time
    
    - ``T_bseq`` -- length of sequence in each batch
    (must divide `t_seq`) i.e. local time
    
    - ``K`` -- count of sequences in each batch
    
    Return:
    containing batchs tensor of shape:
    T_seq/T_bseq, T_bseq, K
    '''
    z_prev = z0 * torch.ones(K)
    
    x_batch = torch.tensor([])
    z_batch = torch.tensor([])
    transition = torch.tensor([0.1, 0.7])
    emission = torch.tensor([0.9, 0.6])

    for i in range(int(T_seq/T_bseq)):
        zs = torch.tensor([])
        xs = torch.tensor([])
        with pyro.plate("plate0", K):
            for t in range(T_bseq):
                # j = K - i % K
                # z.shape == K
                ip_transition = Vindex(transition)[z_prev.type(torch.long)]
                z = pyro.sample("z_%d" % t,
                                pdist.Binomial(1, ip_transition))
                               
                z_prev = z
                # print(ip_transition)
                # print(z)
                ip_emission = Vindex(emission)[z.type(torch.long)]
                x = pyro.sample("x_%d" % t,
                                pdist.Binomial(1, ip_emission))
                # zs.shape will be (after for loop) (K, T_bseq)
                zs = torch.cat([zs,
                                torch.unsqueeze(z, 0)], 0)
                xs = torch.cat([xs,
                                torch.unsqueeze(x, 0)], 0)
        
        # z_batch.shape will be (after for loop) (T_seq/T_bseq, )       
        x_batch = torch.cat([x_batch, torch.unsqueeze(xs.T, 0)], 0)
        z_batch = torch.cat([z_batch, torch.unsqueeze(zs.T, 0)], 0)
    
    print("\nz.shape:")
    print(z.shape)
    print("\nzs.shape:")
    print(zs.shape)
    print("\nz_batch.shape:")
    print(z_batch.shape) 
    return(x_batch, z_batch)


def hmm_quess(z0=None, xs_batch=None, threshold=None):
    '''
    - ``t_seq`` -- general sequence length
    i.e. global time
    
    - ``T_bseq`` -- length of sequence in each batch
    (must divide `t_seq`) i.e. local time
    
    - ``K`` -- count of sequences in each batch
    
    Return:
    containing batchs tensor of shape:
    T_seq/T_bseq, T_bseq, K
    '''
    T = xs_batch.shape[1]
    z_prev = pyro.sample("z0", pdist.Bernoulli(torch.tensor(0.5)))
    # z_prev = z0[:]
    
    x_batch = torch.tensor([])
    z_batch = torch.tensor([])
    
    transition = torch.tensor([0.3, 0.7])
    emission = torch.tensor([0.2, 0.9])
    '''
    pt0 = pyro.sample("pt0", pdist.Uniform(0, 1))
    pt1 = pyro.sample("pt1", pdist.Uniform(0, 1))
    pe0 = pyro.sample("pe0", pdist.Uniform(0, 1))
    pe1 = pyro.sample("pe1", pdist.Uniform(0, 1))

    transition = torch.tensor([pt0, pt1])
    emission = torch.tensor([pe0, pe1])
    '''
    # dist is fake, only obs important:
    pyro.sample("threshold", pdist.Uniform(0, torch.tensor(threshold)),
                obs=torch.tensor(threshold))
    pyro.sample("N", pdist.Uniform(0, torch.tensor(T)),
                obs=torch.tensor(T))
   
    ws = torch.tensor([])

    with pyro.plate("plate0", xs_batch.shape[0]):
        for t in range(T):
            # j = K - i % K
            # z.shape == K
            ip_transition = Vindex(transition)[z_prev.type(torch.long)]
            z = pyro.sample("z_%d" % t,
                            pdist.Bernoulli(ip_transition))

            z_prev = z
            # print(ip_transition)
            # print(z)
            ip_emission = Vindex(emission)[z.type(torch.long)]
            dist_x = pdist.Bernoulli(ip_emission)
            # dist_x = pdist.Binomial(1, ip_emission)
            x = pyro.sample("x_%d" % t,
                            dist_x,)  # obs=xs_batch[:, t])

            # ws = torch.cat([ws,
            #                 torch.unsqueeze(dist_x.log_prob(x), 0)], 0)
            x_obs = pyro.sample("x_obs_%d" % t,
                                dist_x, obs=xs_batch[:, t])
    return(x)


def hmm_fcond0(trace):
    
    N = trace.nodes['N']['value']
    threshold = trace.nodes['threshold']['value']

    # assuming that x_i only has value 0 or 1:
    a = torch.tensor([-1 if trace.nodes['x_%d' % i]['value'] == 0 else 1
                      for i in range(N)])
    b = torch.tensor([-1 if trace.nodes['x_obs_%d' % i]['value'] == 0 else 1
                      for i in range(N)])
    # print("a:")
    # print(a)
    # print("b:")
    # print(b)
    # print("(a*b).sum(): ", (a*b).sum())
    # print("N: ", N)
    
    return((a*b).sum()/float(N) >= threshold)


def hmm_fcond1(trace):
    
    a = [trace.nodes['x_%d' % i]['value']
         for i in range(10)]
    b = [trace.nodes['x_obs_%d' % i]['value']
         for i in range(10)]
    
    return(all([a[i].mean()-b[i].mean() <= (a[i].var()+b[i].var())/2
                for i in range(10)]))


def test1():
    xs_batch, zs_batch = hmm_target(1, 1800, 10, 20)
    
    print("xs_batch.shape:")
    print(xs_batch.shape)

    x = hmm_quess(zs_batch[0, :, 0], xs_batch[0], 2)
    print("x.shape")
    print(x.shape)

    # test forward for obs = [1, 1]
    T = 2
    # T = 10
    # 2
    z0 = torch.tensor([0])
    x0 = torch.zeros((1, T))

    # obs = [1, 1]
    x0[0][-2:] = 1
    # for T=10 x0[0][-5:] = 1
    print("goal x:")
    print(x0)

    x_last = hmm_quess(z0, x0, 2)
    print("last x in seq:")
    print(x_last)
    # trace = pytro.poutine.trace(hmm_quess).get_trace(z0, x0)

    import sim3
    threshold = 1
    # for T=10 threshold = 0.7
    # effective values for seq of len 2 is -1, 0, 1
    # for 10 [-1, -0.9, ... 0, ... 1]
    samples = sim3.rejection_sampling1(
        hmm_quess,
        {"z0": z0, "xs_batch": x0, "threshold": threshold},
        hmm_fcond0, 1000)
    # result: [0.11460922 0.88539078]
    # accurate: [0.18181818, 0.81818182]
    '''
    samples = sim3.rejection_sampling1(
        hmm_quess,
        {"z0": zs_batch[0, :, 0], "xs_batch": xs_batch[0]},
        hmm_fcond1, 100)
    '''
    
    print("len(samples): ", len(samples))
    
    idx = int(samples[-1].nodes['N']['value'])-1
    print(samples[-1].nodes["z_%d" % idx]['value'])
    sim3.plot_results1(samples, var_name="z_%d" % idx)
    

if __name__ == '__main__':
    
    run_m1(N=9000, obs=[1., 0.], delta=100, plot_zs=True)
    # result: 0.4134720700985761, 0.5865279299014239

    # accurate p(z0|x1==1, x2==0): 0.418892  0.581108
    # run_m1(N=3000, obs=[1., 0.], delta=100, plot_zs=True)
    # result p(z0|x1==1, x2==0): 0.45318352059925093, 0.5468164794007491
    

    # run_m1(N=3000, obs=[1.], delta=100, plot_zs=True)
