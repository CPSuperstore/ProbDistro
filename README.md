# ProbDistro
A Python package for handling various continuous and discrete random variables, including normal/gauss, binomial, poisson, and more!

**Contents:**
- [Available Distributions](#available-distributions)
  - [Discrete Distribution](#discrete)
  - [Continuous Distribution](#continuous)
- [Usage](#usage)
  - [Discrete Distribution](#discrete-distribution)
  - [Continuous Distribution](#continuous-distribution)
  - [Converting Between Distributions](#converting-between-distributions)
- [Random Variables](#random-variables)
  - [Jointly Distributed Random Variables](#jointly-distributed-random-variables)
- [Bug Tracker](#bug-tracker)

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
A discrete random variable which can be used to represent 2 situations.
     
1. `include_success_trial = True` - the number of trials which are needed to obtain exactly 1 success. Supports set {1, 2, 3, 4, ...}.
For example, if the probability is 0.25 for an x value of 3, this indicates there is a 25% chance of receiving
no successes until the third trial where the first success is obtained
3. `include_success_trial = False` - the number of failures before the first success
        supports set {0, 1, 2, 3, ...}

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

#### cdf(x: float) -> float:
Returns the value of the cumulative distribution function at `x`

#### pmf(x: float) -> float:
Returns the value of the probability mass function at `x`

#### equals(x: float) -> float:
An alias for the `pmf` method

#### expected_value() -> float:
Returns the expected value of the distribution

#### variance() -> float:
Returns the variance of the distribution

#### less_than(x: float) -> float:
Returns the probability of being less than `x` 

#### less_than_equals(x: float) -> float:
Returns the probability of being less than or equal to `x` 

#### greater_than(x: float) -> float:
Returns the probability of being greater than `x` 

#### greater_than_equals(x: float) -> float:
Returns the probability of being greater than or equal to `x` 

#### mean() -> float:
Returns the mean value of the distribution. This is an alias for the `expected_value` method.

#### standard_deviation() -> float:
Returns the standard deviation of the distribution.

#### between(upper: float, lower: float):
Returns the probability of being between the upper and lower bounds. Entering the upper bound as the lower bound will result in a negative probability.

#### cdf_range(x: typing.Iterable):
Returns a list containing the value of the CDF at every value in the provided iterable (list, range, set, tuple, etc.)

#### pmf_range(x: typing.Iterable):
Returns a list containing the value of the PMF at every value in the provided iterable (list, range, set, tuple, etc.)

#### to_discrete_random_variable(start: float = None, stop: float = None, step: float = None) -> DiscreteRandomVariable:
Returns a discrete random variable representation of this distribution. `Start` is the first value to be included, and `stop` is the last. Intervals of `step` are used.

Note that depending on the distribution, you may not need to specify all three. 
In all cases, the default step is 1, and the start will be the first value in the range of supported values.

For distributions such as Binomial or Hypergeometric, the range of x values has an upper bound, and so that upper limit is the default for `stop`

For distributions such as Geometric or Poisson, x is unbound so you will need to specify the upper limit.

In all cases, the `correct_probabilities` class method is used during conversion, so the probabilities do not necessarily need to add up to 1 (which will be the case with an unbound random variable)

### Continuous Distribution
All continuous distributions extend the `BaseDiscreteDistribution` class. This enforces each discrete distribution to implement the following methods:

#### cdf(self, x: float) -> float:
Returns the value of the cumulative distribution function at `x`

#### pdf(self, x: float) -> float:
Returns the value of the probability density function at `x`

#### equals(self, x: float) -> float:
Returns 0

#### expected_value(self) -> float:
Returns the expected value of the distribution

#### variance(self) -> float:
Returns the variance of the distribution

#### less_than(self, x: float) -> float:
Returns the probability of being less than `x` 

#### less_than_equals(self, x: float) -> float:
Returns the probability of being less than or equal to `x` 

#### greater_than(self, x: float) -> float:
Returns the probability of being greater than `x` 

#### greater_than_equals(self, x: float) -> float:
Returns the probability of being greater than or equal to `x` 

#### mean(self) -> float:
Returns the mean value of the distribution. This is an alias for the `expected_value` method.

#### standard_deviation(self) -> float:
Returns the standard deviation of the distribution.

#### between(self, upper: float, lower: float):
Returns the probability of being between the upper and lower bounds. Entering the upper bound as the lower bound will result in a negative probability.

#### cdf_range(self, x: typing.Iterable):
Returns a list containing the value of the CDF at every value in the provided iterable (list, range, set, tuple, etc.)

#### pdf_range(self, x: typing.Iterable):
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

## Random Variables
ProbDistro also supports random variables. This can be thought of as a table, like so:

| x      | 1   | 2    | 3     | 4     |
|--------|-----|------|-------|-------|
| p(X=x) | 0.5 | 0.25 | 0.125 | 0.125 |

Where the first row contains every possible outcome of the random variable, 
and p(X=x) is the probability that each of those events occur. 
By law of total probability, all the probabilities MUST add up to 1.

This table should be thought of as the PMF as a discrete random variable, where the probability of 1 is 0.5 
and the probability of 4 is 0.125.

To create a random variable which represents the above table, the following code can be used:

```python
import ProbDistro

rv = ProbDistro.DiscreteRandomVariable([1, 2, 3, 4], [0.5, 0.25, 0.125, 0.125])
```

Note that the values of x must be ordered from smallest to largest, and the probabilities must be arranged to match the order of the x values.

The `DiscreteRandomVariable` also extends the `BaseDiscreteDistribution` class, so it implements each of the methods 
listed under [Discrete Distribution](#discrete-distribution) (including `to_discrete_random_variable` which allows you to 
create a subset of this distribution of x values between `start` and `stop`) 

If the probabilities do not add up to 1 (rounded to 10 digits to account for floating point errors), the 
`correct_probabilities` class method can be used to correct the probabilities and make their sum equal 1:

```python
import ProbDistro

rv = ProbDistro.DiscreteRandomVariable.correct_probabilities([1, 2, 3], [0.5, 0.25, 0.125])

print(rv)
```

Output:
```
<DiscreteRandomVariableDistribution x=[1, 2, 3] px=[0.5714285714285714, 0.2857142857142857, 0.14285714285714285]>
```

Finally, if the data is in a dictionary, the `from_dict` class method can be used:

```python
import ProbDistro

rv = ProbDistro.DiscreteRandomVariable.from_dict({1: 0.5, 2: 0.25, 3: 0.125, 4: 0.125})
```

Or if the probabilities do not add up to 1:
```python
import ProbDistro

rv = ProbDistro.DiscreteRandomVariable.from_dict({1: 0.5, 2: 0.25, 3: 0.125}, correct=True)
```

### Jointly Distributed Random Variables
If you have 2 or more discrete random variables, ProbDistro allows you to perform operations on them as if there were a joint distribution. For example:

```python
from ProbDistro import DiscreteRandomVariable

a = DiscreteRandomVariable([1, 2, 3, 4], [0.5, 0.25, 0.125, 0.125])
b = DiscreteRandomVariable([2, 4, 6], [0.3, 0.6, 0.1])

# returns P(a=1 and b=4)
print(a.p_and(a(1), b(4)))

# returns P(a=1 or b=4)
print(a.p_or(a(1), b(4)))

# returns P(a=1 or b=4), assuming a and b are disjoint
print(a.p_disjoint_or(a(1), b(4)))

# returns P(a=1 | b=4)
print(a.p_given(a(1), b(4)))
```

Note that each of these operations are static methods, so the following code is equally valid
```python
from ProbDistro import DiscreteRandomVariable

a = DiscreteRandomVariable([1, 2, 3, 4], [0.5, 0.25, 0.125, 0.125])
b = DiscreteRandomVariable([2, 4, 6], [0.3, 0.6, 0.1])

# returns P(a=1 and b=4)
print(DiscreteRandomVariable.p_and(a(1), b(4)))

# returns P(a=1 or b=4)
print(DiscreteRandomVariable.p_or(a(1), b(4)))

# returns P(a=1 or b=4), assuming a and b are disjoint
print(DiscreteRandomVariable.p_disjoint_or(a(1), b(4)))

# returns P(a=1 | b=4)
print(DiscreteRandomVariable.p_given(a(1), b(4)))
```

This is not limited to `a=1` and `b=4`. The `less_than`, `greater_than` and other methods of that nature can also be used in place of a(1) (or a.equals(1)):

```python
from ProbDistro import DiscreteRandomVariable

a = DiscreteRandomVariable([1, 2, 3, 4], [0.5, 0.25, 0.125, 0.125])
b = DiscreteRandomVariable([2, 4, 6], [0.3, 0.6, 0.1])

# returns P(a > 1 and b<=4)
print(DiscreteRandomVariable.p_and(a.greater_than(1), b.less_than_equals(4)))
```

For improved readability, the following methods are also available, which are aliases of the above methods:
 - `intersection` - Alias for`p_and`
 - `union` - Alias for `p_or`
 - `disjoint_union` - Alias for `p_disjoint_or`

Additionally, there exist other operations of jointly distributed random variables:

#### Joint Distribution Table
To generate a joint distribution table, the following code is used:
```python
from ProbDistro import DiscreteRandomVariable

a = DiscreteRandomVariable([1, 2, 3, 4], [0.5, 0.25, 0.125, 0.125])
b = DiscreteRandomVariable([2, 4, 6], [0.3, 0.6, 0.1])

a.jointly_distributed_table(b)
```

Which returns the following 2d list:
```
[
  [0.15, 0.075, 0.0375, 0.0375], 
  [0.3,  0.15,  0.075,  0.075 ], 
  [0.05, 0.025, 0.0125, 0.0125]
]
```

Where the values of `a` are placed along the top, and values of `b` are moving downwards.
The array can be interpreted as the following table:

| b \ a | 1    | 2     | 3      | 4      |
|-------|------|-------|--------|--------|
| 2     | 0.15 | 0.075 | 0.0375 | 0.0375 |
| 4     | 0.3  | 0.15  | 0.075  | 0.075  |
| 6     | 0.05 | 0.025 | 0.0125 | 0.0125 |

And if we were to add the cells containing the marginal probabilities, we can restore the original probabilities given to the constructor:

| b \ a  | 1    | 2     | 3      | 4      | p(B=b) |
|--------|------|-------|--------|--------|--------|
| 2      | 0.15 | 0.075 | 0.0375 | 0.0375 | 0.3    |
| 4      | 0.3  | 0.15  | 0.075  | 0.075  | 0.6    |
| 6      | 0.05 | 0.025 | 0.0125 | 0.0125 | 0.1    |
| p(A=a) | 0.5  | 0.25  | 0.125  | 0.125  | 1      |

#### Covariance
The covariance can be calculated through the following command:

```python
from ProbDistro import DiscreteRandomVariable

a = DiscreteRandomVariable([1, 2], [0.4, 0.6])
b = DiscreteRandomVariable([2, 3, 4], [0.3, 0.2, 0.5])

print(a.covariance(b))
```

Output:
```
1.7763568394002505e-15
```

#### Correlation
The correlation can be calculated through the following command:

```python
from ProbDistro import DiscreteRandomVariable

a = DiscreteRandomVariable([1, 2], [0.4, 0.6])
b = DiscreteRandomVariable([2, 3, 4], [0.3, 0.2, 0.5])

print(a.correlation(b))
```

Output:
```
1.8809278223539317e-15
```

## Bug Tracker
To report bugs or leave feedback, please visit our bug tracker at 
https://github.com/CPSuperstore/ProbDistro/issues

Thanks for using ProbDistro
