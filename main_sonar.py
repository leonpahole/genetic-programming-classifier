import random
import operator
import csv
import itertools

import numpy
from sklearn.metrics import f1_score, accuracy_score

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import pygraphviz as pgv
from numpy import arange

import sys

MEASUREMENTS_AMOUNT = 60


def parse_element_in_row(e):
    if e is 'R':
        return True
    elif e is 'M':
        return False

    return float(e)


# read sonar data
with open("sonar.all-data") as sonar_csv:
    sonar_csv_reader = csv.reader(sonar_csv)
    sonar_data = list(list(parse_element_in_row(elem) for elem in row) for row in sonar_csv_reader)

# "IN" -> prefix of the entry
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, MEASUREMENTS_AMOUNT), bool, "IN")

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)


# floating point operators
def protected_division(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protected_division, [float, float], float)


# logic operators
def if_then_else(ipt, output1, output2):
    if ipt:
        return output1
    else:
        return output2


pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

# 1.0 -> we are maximizing the fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# individual: genetic programming tree
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# generate sometimes full tree (all leafs same depth) and sometimes half tree (not all leafs same depth)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)

# generator of one individual
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

# generator of whole population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile program into callable function
toolbox.register("compile", gp.compile, pset=pset)

SAMPLE_AMOUNT_FOR_EVALUATION = 200


# evaluate one inidividual in population -> F-score
def evaluate_individual(individual):
    # compile to callable function
    func = toolbox.compile(expr=individual)

    # sample random data
    sonar_sample_data = random.sample(sonar_data, SAMPLE_AMOUNT_FOR_EVALUATION)

    # prepare actual expected values
    y_true = [x[MEASUREMENTS_AMOUNT] for x in sonar_sample_data]

    # predict using the function
    y_pred = [bool(func(*row[:MEASUREMENTS_AMOUNT])) for row in sonar_sample_data]

    # use F1 score to evaluate the individual
    return f1_score(y_true, y_pred),


# evaluate on entire database
def evaluate_individual_final(individual):
    # compile to callable function
    func = toolbox.compile(expr=individual)

    # prepare actual expected values
    y_true = [x[MEASUREMENTS_AMOUNT] for x in sonar_data]

    # predict using the function
    y_pred = [bool(func(*row[:MEASUREMENTS_AMOUNT])) for row in sonar_data]

    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred),


toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=50))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=50))


def main(tournament_size, p_crossover, p_mutate, generation_count, population_size):
    random.seed(10)

    try:
        toolbox.unregister("select")
    except AttributeError:
        pass

    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    population = toolbox.population(n=population_size)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(population, toolbox, p_crossover, p_mutate, generation_count, stats, halloffame=hall_of_fame)

    return population, stats, hall_of_fame


if __name__ == "__main__":

    if len(sys.argv) > 1:

        population_size = 100
        generation_count = 100

        tournament_size_min, tournament_size_max, tournament_size_step = 2, 10, 2
        p_crossover_min, p_crossover_max, p_crossover_step = 0.1, 0.5, 0.1
        p_mutate_min, p_mutate_max, p_mutate_step = 0.1, 0.5, 0.1

        best_combination = None
        best_score = None
        best_individual = None

        counter = 1

        for ts in arange(tournament_size_min, tournament_size_max, tournament_size_step):
            for pc in arange(p_crossover_min, p_crossover_max, p_crossover_step):
                for pm in arange(p_mutate_min, p_mutate_max, p_mutate_step):
                    pop, stats, hof = main(ts, pc, pm, generation_count, population_size)
                    best_f_score = evaluate_individual(hof.items[-1])

                    print('Counter:', str(counter))
                    print('Combination: ts=', str(ts), ', pc=', str(pc), ', pm=', str(pm))
                    print('Best f score: ', str(best_f_score))

                    if best_score is None or best_f_score > best_score:
                        best_score = best_f_score
                        best_combination = (ts, pc, pm)
                        best_individual = hof.items[-1]

                    counter += 1
    else:
        population_size = 100
        generation_count = 1000

        pop, stats, hof = main(8, 0.3, 0.4, generation_count, population_size)
        best_individual = hof.items[-1]
        best_score = evaluate_individual_final(best_individual)
        best_combination = None

    print('Best individual')
    print(best_combination)
    print('Score (a, f): ', str(best_score))

    nodes, edges, labels = gp.graph(best_individual)

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("best_individual.pdf")
