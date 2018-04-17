# Simple Genetic Algorithm

A simple genetic algorithm that solves the one-max problem.

### Initialization
A population N of randomly generated binary strings of the specified size n. The strings are generated with the probability of each bit being 1 of 0.5.

### Selection
Parents are selected using binary tournament selection of k=2. So, given a population of size N, a random individual is selected. Then another random individual is selected. The more fit individual is selected as a parent. This procedure is repeated for the second parent.

### Recombination
There is a 60% chance that uniform crossover will be applied to those parents to produce two new children for the 'next generation'. Uniform crossover generates the first child by randomly picking from which parent to get each bit. Then, the second child is the inverse (opposite) of the first. Otherwise, there is a 40% chance that the parents will be kept in the 'new generation'.

### Replacement
For a population of size N, N-2 children are generated. All but the best 2 parents are replaced with this set of children. This is called elitism replacement becuase we are keeping the very best individuals around each generation.

### Fitness Function
The fitness function is ***onemax***, which is the sum of 1s in the binary string.

### Finding the Minimum Population Size
The population size starts at size 10. Then, the GA is ran 5 times and each time we check to see if the global optimum is present in the new generation. If the global optimum is not present in each of the 5 generations, then we then double the population size and repeat the above procedure. Once we find the smallest population size where the global optimum is present in each of the 5 generations we have found an upper bound. Next, we take the midpoint between this upper-bound and the lower-bound where it last failed. We repeat this procedure until we find a minimum population size. This procedure of finding the minimum population size to solve a problem is called ***bisection***.

### Minimum Population Sizes for Different String Sizes
| String Size |  Run 1  |  Run 2  |  Run 3  |  Avg   |
|:-----------:|:-------:|:-------:|:-------:|:------:|
| 20          | 11      | 7       | 13      | 10.33  |
| 50          | 21      | 19      | 17      | 19.00  |


#### To run this program:
```
python3.6 main.py
```
