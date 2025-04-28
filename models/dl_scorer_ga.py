import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.dl_scorer import build_scoring_model
from data.dataloader import load_data

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("lstm_units", random.randint, 32, 128)
toolbox.register("dropout_rate", random.uniform, 0.2, 0.6)
toolbox.register("lr", random.uniform, 0.0001, 0.01)
toolbox.register("optimizer_idx", random.randint, 0, 2)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.lstm_units, toolbox.dropout_rate, toolbox.lr, toolbox.optimizer_idx), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_individual(individual):
    lstm_units, dropout_rate, lr, optimizer_idx = individual
    optimizer_name = ['adam', 'rmsprop', 'sgd'][int(optimizer_idx)]

    model = build_scoring_model(
        vocab_size=10000,
        input_length=100,
        lstm_units=int(lstm_units),
        dropout_rate=float(dropout_rate),
        lr=float(lr),
        optimizer_name=optimizer_name
    )

    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
    preds = model.predict(X_val)
    preds = (preds > 0.5).astype(int)

    return (accuracy_score(y_val, preds),)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga():
    population = toolbox.population(n=10)
    NGEN = 5

    for gen in range(NGEN):
        print(f"\n-- Generation {gen} --")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    
    top_ind = tools.selBest(population, k=1)[0]
    print("\n✅ Best individual:", top_ind)
    print("📈 Best accuracy:", top_ind.fitness.values[0])
    
    best = tools.selBest(population, k=1)[0]
    print("\n✅ Best individual:", best)
    return {
        "lstm_units": int(best[0]),
        "dropout_rate": float(best[1]),
        "lr": float(best[2]),
        "optimizer_name": ['adam', 'rmsprop', 'sgd'][int(best[3])]
    }
