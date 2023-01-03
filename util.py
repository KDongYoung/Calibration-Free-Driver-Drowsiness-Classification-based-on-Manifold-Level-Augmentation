import torch
import torch.nn as nn
import random
import numpy as np

'''
##################################################################################################################
# MixStyle Augmentation
##################################################################################################################
'''
class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random',batch_size=1, num_domain=1):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

        self.batch_size=batch_size
        self.num_domain=num_domain

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2], keepdim=True) # [1, 2], [2], [1]
        var = x.var(dim=[2], keepdim=True) # [1, 2], [2], [1]
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B-1, -1, -1) 
            perm_b, perm_a = perm.chunk(2) 

            if len(perm_a)<len(perm_b):
                perm_b = perm_b[torch.randperm(B // 2+1)]
                perm_a = perm_a[torch.randperm(B // 2)]
            elif len(perm_a)<len(perm_b):
                perm_b = perm_b[torch.randperm(B // 2)]
                perm_a = perm_a[torch.randperm(B // 2+1)]
            elif len(perm_a)==len(perm_b):
                perm_b = perm_b[torch.randperm(B // 2)]
                perm_a = perm_a[torch.randperm(B // 2)]

            perm = torch.cat([perm_b, perm_a], 0)

        elif self.mix=="random_shift":
            shift_num=random.randint(1,self.num_domain-1)
            perm=torch.arange(0,B,dtype=torch.long)
            perm_a=perm[(-1)*self.batch_size*shift_num:]
            perm_b=perm[:(-1)*self.batch_size*shift_num]
            perm=torch.cat([perm_a,perm_b],0)
        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix


'''
##################################################################################################################
# Mixup Augmentation
##################################################################################################################
'''

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :] # mix x and randomly shuffled x
    mixed_y = lam * y + (1-lam) * y[index] # y_a: target, y_b: target of randomly shuffled x
    return mixed_x, mixed_y # mixed_x, mixed_target

'''
##################################################################################################################
# One hot encoding
##################################################################################################################
'''
def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes) 
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return y_onehot