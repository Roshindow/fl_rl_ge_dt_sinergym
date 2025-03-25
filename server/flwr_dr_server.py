import pickle
import argparse
import deap
import os
import flwr as fl
from logging import INFO
from flwr.server.strategy import FedAvg
from flwr.common import Parameters
from flwr.common.logger import log

from numpy import random

class GeneticStrategy(FedAvg):
    def __init__(self, path='log.txt', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_genotype = None
        self.best_phenotype = None

    def aggregate_fit(self, rnd, results, failures):
        # Extract genotypes and their fitnesses from the results
        genotypes = [pickle.loads(res.parameters.tensors[0])[0] for _, res in results]
        phenotypes = [pickle.loads(res.parameters.tensors[0])[1] for _, res in results]
        fitness_scores = [genotype.fitness.values[0] for genotype in genotypes]
        
        # Find the best genotype in the current round
        max_fitness = max(fitness_scores)
        best_current_genotype = genotypes[fitness_scores.index(max_fitness)]
        best_current_phenotype = phenotypes[fitness_scores.index(max_fitness)]
        
        # Compare with the best across generations
        if self.best_genotype is None or max_fitness > self.best_genotype.fitness.values[0]:
            log(
                INFO,
                "====== new best genotype: %s ======",
                best_current_genotype,
            )
            self.best_genotype = best_current_genotype
            if self.best_phenotype is None or self.best_phenotype != best_current_phenotype:
                log(
                    INFO,
                    "====== new best phenotype: %s ======",
                    best_current_phenotype,
                )
                self.best_phenotype = best_current_phenotype
        
        # Serialize new_genotype into a Parameters object
        serialized_type = pickle.dumps([self.best_genotype,self.best_phenotype]) #nonfl test
        aggregated_parameters = Parameters(tensors=[serialized_type], tensor_type="DEAP_Individual")

        # Metrics (example: best fitness)
        metrics = {
            "best_fitness": self.best_genotype.fitness.values[0],
            "round": rnd,
        }
        log(
            INFO,
            "====== best score: %s ======",
            self.best_genotype.fitness.values[0],
        )
        
        # LOG RESULTS
        #with open(resfile, "a") as f:
            #f.write(f"run best:{max_fitness}; genotype:{best_current_genotype}\n top best:{self.best_genotype.fitness.values[0]}; genotype:{self.best_genotype}\n")
        
        return aggregated_parameters, metrics

parser = argparse.ArgumentParser()
#parser.add_argument("--address", default="flower-server:8080", type=str, help="The server address")
parser.add_argument("--address", default="0.0.0.0:8080", type=str, help="The server address")
parser.add_argument("--rounds", default=1, type=int, help="The number of rounds")
parser.add_argument("--min_fit_clients", default=1, type=int, help="The minimum number of clients required for the fit phase")
parser.add_argument("--min_eval_clients", default=1, type=int, help="The minimum number of clients required for the evaluation phase")
parser.add_argument("--min_available_clients", default=1, type=int, help="The minimum number of clients required")
args = parser.parse_args()

resdir = f"/s_data"
resfile = os.path.join(resdir, "res.txt")

if __name__ == "__main__":
    strategy = GeneticStrategy(
        path=resfile,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_eval_clients,
        min_available_clients=args.min_available_clients
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(
            num_rounds=args.rounds
        ),
        strategy=strategy
    )
