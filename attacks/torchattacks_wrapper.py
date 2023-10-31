import torchattacks

def bim(model, eps=8/255, alpha=2/255, nb_iters=10, **kwargs):
    attack = torchattacks.BIM(model, eps=eps, alpha=alpha, steps=nb_iters)
    return attack


def apgd(model, norm='Linf', eps=8/255, nb_iters=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False, **kwargs):
    attack = torchattacks.APGD(model, norm=norm, eps=eps, steps=nb_iters, n_restarts=n_restarts, seed=seed, loss=loss, eot_iter=eot_iter, rho=rho, verbose=verbose)
    return attack

def difgsm(model, eps=8/255, alpha=2/255, nb_iters=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False, **kwargs):
    attack = torchattacks.DIFGSM(model, eps=8/255, alpha=2/255, steps=nb_iters, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
    return attack

def eaden(model, kappa=0, lr=0.01, nb_iters=100, **kwargs):
    attack = torchattacks.EADEN(model, kappa=kappa, lr=lr, max_iterations=nb_iters)
    return attack

def eadl1(model, kappa=0, lr=0.01, nb_iters=100, **kwargs):
    attack = torchattacks.EADL1(model, kappa=kappa, lr=lr, max_iterations=nb_iters)
    return attack