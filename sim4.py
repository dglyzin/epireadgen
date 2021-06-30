import torch
import pyro
import pyro.distributions as pdist
from pyro.ops.indexing import Vindex


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


def hmm_quess(z0=None, xs_batch=None):
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
    z_prev = z0[:]
    
    x_batch = torch.tensor([])
    z_batch = torch.tensor([])

    pt0 = pyro.sample("pt0", pdist.Uniform(0, 1))
    pt1 = pyro.sample("pt1", pdist.Uniform(0, 1))
    pe0 = pyro.sample("pe0", pdist.Uniform(0, 1))
    pe1 = pyro.sample("pe1", pdist.Uniform(0, 1))

    transition = torch.tensor([pt0, pt1])
    emission = torch.tensor([pe0, pe1])

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
            dist_x = pdist.Binomial(1, ip_emission)
            x = pyro.sample("x_%d" % t,
                            dist_x,)  # obs=xs_batch[:, t])

            # ws = torch.cat([ws,
            #                 torch.unsqueeze(dist_x.log_prob(x), 0)], 0)
            x_obs = pyro.sample("x_obs_%d" % t,
                                dist_x, obs=xs_batch[:, t])
    return(x)


def hmm_fcond0(trace):
    # assuming that x_i only has value 0 or 1:
    a = torch.tensor([-1 if trace.nodes['x_%d' % y]['value']==0 else 1
                      for i in range(10)])
    b = torch.tesnor([-1 if trace.nodes['x_obs_%d' % y]['value']==0 else 1
                      for i in range(10)])
    
    return((a*b).sum())


def hmm_fcond1(trace):
    
    a = [trace.nodes['x_%d' % i]['value']
         for i in range(10)]
    b = [trace.nodes['x_obs_%d' % i]['value']
         for i in range(10)]
    
    return(all([a[i].mean()-b[i].mean() <= (a[i].var()+b[i].var())/2
                for i in range(10)]))


if __name__ == '__main__':
    xs_batch, zs_batch = hmm_target(1, 1800, 10, 20)
    
    x = hmm_quess(zs_batch[0, :, 0], xs_batch[0])
    print(x.shape)

    import sim3
    samples = sim3.rejection_sampling(
        hmm_quess,
        {"z0": zs_batch[0, :, 0], "xs_batch": xs_batch[0]},
        hmm_fcond1, 100)

    print("len(samples): ", len(samples))
