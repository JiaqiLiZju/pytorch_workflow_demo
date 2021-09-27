def plot_mixture_density(weights, norm_params, n=10000):
    ''' Weight of each component'''
    import numpy as np
    import numpy.random
    import scipy.stats as ss
    import matplotlib.pyplot as plt
    
    # Parameters of the mixture components
    n_components = norm_params.shape[0]
    
    # A stream of indices from which to choose the component
    mixture_idx = numpy.random.choice(len(weights), size=n, replace=True, p=weights)
    
    # y is the mixture sample
    y = numpy.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                       dtype=np.float64)

    # Theoretical PDF plotting -- generate the x and y plotting positions
    xs = np.linspace(y.min(), y.max(), 200)
    ys = np.zeros_like(xs)

    for (l, s), w in zip(norm_params, weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w

    sns.distplot(y, hist=False)

#     plt.plot(xs, ys)
#     plt.hist(y, bins="fd")
#     plt.xlabel("x")
#     plt.ylabel("f(x)")
#     plt.show()