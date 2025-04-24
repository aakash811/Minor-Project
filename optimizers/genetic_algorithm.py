from deap import base, creator, tools, algorithms
import random
from models.lstm_model import build_model_from_genome

EMBEDDING_DIM = [64, 128, 256]
LSTM_UNITS = [32, 64, 128]
DROPOUT_RATES = [0.1, 0.2, 0.3]
OPTIMIZERS = list(range(3))

def evaluate(individual, x_train, y_train, x_val, y_val, vocab_size, input_length):
    model = build_model_from_genome(vocab_size, input_length, individual)
    model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)
    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    return (acc,)

def setup_ga(x_train, y_train, x_val, y_val, vocab_size, input_length, n_gen=5, pop_size=6):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("embedding_dim", lambda: random.choice(EMBEDDING_DIM))
    toolbox.register("lstm_units", lambda: random.choice(LSTM_UNITS))
    toolbox.register("dropout_rate", lambda: random.choice(DROPOUT_RATES))
    toolbox.register("optimizer_idx", lambda: random.choice(OPTIMIZERS))

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.embedding_dim, toolbox.lstm_units,
                      toolbox.dropout_rate, toolbox.optimizer_idx), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("evaluate", evaluate, x_train=x_train, y_train=y_train,
                     x_val=x_val, y_val=y_val, vocab_size=vocab_size, input_length=input_length)

    population = toolbox.population(n=pop_size)
    best_individual = None

    for gen in range(n_gen):
        print(f"Generation {gen}")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.3)
        fits = list(toolbox.map(toolbox.evaluate, offspring))

        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=pop_size)
        best_individual = tools.selBest(population, k=1)[0]
        print(f"Best accuracy: {best_individual.fitness.values[0]} with {best_individual}")

    return best_individual
