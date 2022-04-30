# ProbDistro
A Python package for handling various continuous and discrete random variables, including normal/gauss, binomial, poisson, and more!

## Available Distributions
The following is a list of discrete and continuous distributions this library has to offer

### Discrete
A discrete distribution is when there are countably many outcomes (example, any integer of 0 or higher, or any integer between 5 and 10)

#### Bernoulli
A discrete random variable which can be used to represent the outcome of a single yes/no experiment. 
It has a probability of success "p"

#### Binomial
A discrete random variable which can be used to represent the outcome of "n" yes/no experiments
with "p" probability of success, and "1 - p" (or q) probability of failure

#### Geometric
A discrete random variable which can be used to represent the number of Bernoulli trials 
which are needed to obtain exactly 1 success.

#### Hypergeometric
A discrete random variable which can be used to represent the probability of receiving "x" successes in
"n" dependent draws from a finite population without replacement (draws are not independent of each other)
from a total population of "N" which contains "K" success states

#### Poisson
A discrete random variable which can be used to represent the probability of "x" events occurring in
a given time interval, provided they arrive with a mean rate of "rate", and arrivals are independent

### Continuous
A continuous distribution is when there are uncountably many outcomes (example, any real number of 0 or higher, or any real number between 5 and 10)

#### Exponential
A continuous random variable representing the probability of time between events of a Poisson distribution
(Occurs independently, continuously, and at an average rate)

#### Normal
A continuous random variable representing a bell curve distribution, which is ubiquitous.

#### Uniform
A continuous random variable representing equally likely outcomes between "a" and "b"

## Usage
Simply instantiate the desired distribution, and use its methods to perform the desired calculations.

For example, with a Poisson distribution

```python
import ProbDistro

p = ProbDistro.Poisson(0.75)

# print the probability 5 or fewer events will occur
print(p.less_than_equals(5))

# print the probability exactly 2 events will occur
print(p(5))
```

Or, to convert one distribution to another:

```python
import ProbDistro

p = ProbDistro.Binomial(200, 0.02)

# convert the binomial distribution to normal distribution
n = ProbDistro.conversion.binomial_to_normal(p)

# calculate the value of the normal distro at 5 on the CDF and PDF functions
print(n.cdf(5))
print(n.pdf(5))
```

### Discrete Distribution
All continuous distributions extend the `BaseDiscreteDistribution` class. This enforces each discrete distribution to implement the following methods:

#### `cdf(x: float) -> float:`
Returns the value of the cumulative distribution function at `x`

#### `pmf(x: float) -> float:`
Returns the value of the probability mass function at `x`

#### `equals(x: float) -> float:`
An alias for the `pmf` method

#### `expected_value() -> float:`
Returns the expected value of the distribution

#### `variance() -> float:`
Returns the variance of the distribution

#### `less_than(x: float) -> float:`
Returns the probability of being less than `x` 

#### `less_than_equals(x: float) -> float:`
Returns the probability of being less than or equal to `x` 

#### `greater_than(x: float) -> float:`
Returns the probability of being greater than `x` 

#### `greater_than_equals(x: float) -> float:`
Returns the probability of being greater than or equal to `x` 

#### `mean() -> float:`
Returns the mean value of the distribution. This is an alias for the `expected_value` method.

#### `standard_deviation() -> float:`
Returns the standard deviation of the distribution.

#### `between(upper: float, lower: float):`
Returns the probability of being between the upper and lower bounds. Entering the upper bound as the lower bound will result in a negative probability.

#### `cdf_range(x: typing.Iterable):`
Returns a list containing the value of the CDF at every value in the provided iterable (list, range, set, tuple, etc.)

#### `pmf_range(x: typing.Iterable):`
Returns a list containing the value of the PMF at every value in the provided iterable (list, range, set, tuple, etc.)


### Continuous Distribution
All continuous distributions extend the `BaseDiscreteDistribution` class. This enforces each discrete distribution to implement the following methods:

#### `cdf(self, x: float) -> float:`
Returns the value of the cumulative distribution function at `x`

#### `pdf(self, x: float) -> float:`
Returns the value of the probability density function at `x`

#### `equals(self, x: float) -> float:`
Returns 0

#### `expected_value(self) -> float:`
Returns the expected value of the distribution

#### `variance(self) -> float:`
Returns the variance of the distribution

#### `less_than(self, x: float) -> float:`
Returns the probability of being less than `x` 

#### `less_than_equals(self, x: float) -> float:`
Returns the probability of being less than or equal to `x` 

#### `greater_than(self, x: float) -> float:`
Returns the probability of being greater than `x` 

#### `greater_than_equals(self, x: float) -> float:`
Returns the probability of being greater than or equal to `x` 

#### `mean(self) -> float:`
Returns the mean value of the distribution. This is an alias for the `expected_value` method.

#### `standard_deviation(self) -> float:`
Returns the standard deviation of the distribution.

#### `between(self, upper: float, lower: float):`
Returns the probability of being between the upper and lower bounds. Entering the upper bound as the lower bound will result in a negative probability.

#### `cdf_range(self, x: typing.Iterable):`
Returns a list containing the value of the CDF at every value in the provided iterable (list, range, set, tuple, etc.)

#### `pdf_range(self, x: typing.Iterable):`
Returns a list containing the value of the PDF at every value in the provided iterable (list, range, set, tuple, etc.)

### Converting Between Distributions
ProbDistro includes a set of functions for converting between different types of distributions. 
Each method uses the same signature where the sole parameter is the distribution to be converted, and the return value is the converted target distribution.

For example, converting a binomial distribution (discrete) to a normal distribution (continuous)
```python
import ProbDistro

binomial = ProbDistro.Binomial(200, 0.02)

normal = ProbDistro.conversion.binomial_to_normal(binomial)
```

The following is a list of available conversions:
- binomial to normal
- normal to binomial
- bernoulli to binomial
- binomial to bernoulli
- binomial to poisson
- exponential to poisson
- poisson to exponential
- exponential to geometric
- geometric to exponential

## Bug Tracker
To report bugs or leave feedback, please visit our bug tracker at 
https://github.com/CPSuperstore/ProbDistro/issues

Thanks for using ProbDistro