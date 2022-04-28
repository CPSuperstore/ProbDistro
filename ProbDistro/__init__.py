from ProbDistro.discrete_distributions.bernoulli import Bernoulli
from ProbDistro.discrete_distributions.binomial import Binomial
from ProbDistro.discrete_distributions.geometric import Geometric
from ProbDistro.discrete_distributions.poisson import Poisson
from ProbDistro.discrete_distributions.hypergeometric import Hypergeometric

from ProbDistro.continuous_distributions.uniform import Uniform
from ProbDistro.continuous_distributions.normal import Normal
from ProbDistro.continuous_distributions.exponential import Exponential

from ProbDistro.random_variable.random_variable import RandomVariable

import ProbDistro.conversion as conversions

# aliases for distributions
Pois = Poisson
Bin = Binomial
Geo = Geometric
Ber = Bernoulli
Hyp = Hypergeometric

Uni = Uniform
Norm = Normal
Gaussian = Normal
Gauss = Normal
Laplace_Gauss = Normal
Exp = Exponential
