from deap import base, creator, gp, tools, algorithms
import math, operator, numpy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Because division is scary
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Because log is scary
def protectedLog(x):
    try:
        return math.log(x)
    except ValueError:
        return 1

dataset = [[-1.0, 0.0000],   [-0.9,-0.1629],    [-0.8, -0.2624],
           [-0.7, -0.3129],  [-0.6, -0.3264],   [-0.5, -0.3125],
           [-0.4, -0.2784],  [-0.3, -0.2289],   [-0.2, -0.1664],
           [-0.1, -0.0909],  [0.0, -0.0],       [0.1, 0.1111],
           [0.2, 0.2496],    [0.3, 0.4251],     [0.4, 0.6496],
           [0.5, 0.9375],    [0.6, 1.3056],     [0.7, 1.7731],
           [0.8, 2.3616],    [0.9, 3.0951],     [1.0, 4.0000]]

# Create the primitives set
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(protectedLog, 1)
pset.addPrimitive(math.exp, 1)

# Rename argument to simply x
pset.renameArguments(ARG0='x')

# Define genotype and fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Define the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # MSE
    errors = (abs(func(point[0]) - point[1]) for point in dataset)
    return math.fsum(errors),
    # Squared Errors
    sqerrors = ((func(point[0]) - point[1])**2 for point in dataset)
    return math.fsum(sqerrors) / len(points),


toolbox.register("evaluate", evalSymbReg, points=dataset)
# toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Hold some statistics
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
# mstats = tools.Statistics(lambda ind: ind.fitness.values)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

# Launching the evolution
def evolution():
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.0, 50, stats=mstats,
                                   halloffame=hof, verbose=True)
    return pop, log, hof

pop, log, hof = evolution()

# Print info for best solution found:
best = hof.items[0]
print("-- Best Individual = ", best)
print("-- Best Fitness = ", best.fitness.values[0])

# extract statistics:
# print(log.chapters['fit'].select('min'))
# print(log.chapters['fitness'])
minFitnessValues = log.chapters['fitness'].select("min")
minSizeValues = log.chapters['size'].select("min")
print("MIN FITNESS: ", minFitnessValues)

# # plot statistics:
# sns.set_style("whitegrid")
# plt.plot(minFitnessValues, color='red')
# plt.plot(minSizeValues, color='green')
# red_patch = mpatches.Patch(color='red', label='Best Fitness')
# green_patch = mpatches.Patch(color='green', label='Size')
# plt.legend(handles=[red_patch, green_patch])
# plt.xlabel('Generation')
# plt.ylabel('Best Fitness & Size')
# plt.title('Best fitness over generations')
# plt.show()

# plot statistics:
sns.set_style("whitegrid")
plt.plot(minFitnessValues, color='red')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Fitness of best individual over generations')
plt.show()

plt.plot(minSizeValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Size')
plt.title('Size of best individual over generations')
plt.show()