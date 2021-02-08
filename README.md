# GeneticProgrammingExpression
Python implementation of genetic programming that solves a symbolic expression.

The following graph shows the result a DEAP-based GP with the following parameters:
- Crossover with probability = 0.7
- Mutation with probability = 0.0
- Generations = 50

Final Expression: add(mul(x, exp(x)), mul(mul(mul(x, mul(x, exp(x))), mul(x, x)), sin(sin(exp(mul(mul(mul(x, mul(x, exp(sin(x)))), mul(x, x)), sin(exp(x))))))))

![Best fitness over generations](imgs/GP_1000_MSE.png)
![Size for best generations](imgs/GP_1000_MSE_SIZE.png)
