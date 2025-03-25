import warnings
warnings.filterwarnings("ignore", message="The epw.data submodule is not installed")

from datetime import datetime
import os, sys
import traceback
from typing import Any, Dict, Optional, Union, List, Callable
import re
from dataclasses import dataclass
from enum import Enum

import pickle
import argparse
import string
import deap

import numpy as np

from joblib import parallel_backend, Parallel, delayed

import gymnasium as gym
import sinergym

#from sinergym.utils.callbacks import *
from sinergym.utils.constants import *
from sinergym.utils.logger import WandBOutputFormat, LoggerStorage, TerminalLogger
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *

from dt import EpsGreedyLeaf, PythonDT, RandomlyInitializedEpsGreedyLeaf
from grammatical_evolution import GrammaticalEvolutionTranslator, grammatical_evolution, differential_evolution

import logging
gym.logger.set_level(logging.ERROR)

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*use of fork.*")

def string_to_dict(x):
    """
    This function splits a string into a dict.
    The string must be in the format: key0-value0#key1-value1#...#keyn-valuen
    """
    result = {}
    items = x.split("#")

    for i in items:
        key, value = i.split("-")
        try:
            result[key] = int(value)
        except:
            try:
                result[key] = float(value)
            except:
                result[key] = value

    return result

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", default=1, type=int, help="The number of jobs to use for the evolution")
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument("--environment_name", default="LunarLander-v2", help="The name of the environment in the OpenAI Gym framework")
parser.add_argument("--n_actions", default=4, type=int, help="The number of action that the agent can perform in the environment")
parser.add_argument("--learning_rate", default="auto", help="The learning rate to be used for Q-learning. Default is: 'auto' (1/k)")
parser.add_argument("--df", default=0.9, type=float, help="The discount factor used for Q-learning")
parser.add_argument("--eps", default=0.05, type=float, help="Epsilon parameter for the epsilon greedy Q-learning")
parser.add_argument("--input_space", default=8, type=int, help="Number of inputs given to the agent")
parser.add_argument("--episodes", default=50, type=int, help="Number of episodes that the agent faces in the fitness evaluation phase")
parser.add_argument("--episode_len", default=1000, type=int, help="The max length of an episode in timesteps")
parser.add_argument("--lambda_", default=30, type=int, help="Population size")
parser.add_argument("--generations", default=1000, type=int, help="Number of generations")
parser.add_argument("--cxp", default=0.5, type=float, help="Crossover probability")
parser.add_argument("--mp", default=0.5, type=float, help="Mutation probability")
parser.add_argument("--mutation", default="function-tools.mutUniformInt#low-0#up-40000#indpb-0.1", type=string_to_dict, help="Mutation operator. String in the format function-value#function_param_-value_1... The operators from the DEAP library can be used by setting the function to 'function-tools.<operator_name>'. Default: Uniform Int Mutation")
parser.add_argument("--crossover", default="function-tools.cxOnePoint", type=string_to_dict, help="Crossover operator, see Mutation operator. Default: One point")
parser.add_argument("--selection", default="function-tools.selTournament#tournsize-2", type=string_to_dict, help="Selection operator, see Mutation operator. Default: tournament of size 2")

parser.add_argument("--genotype_len", default=100, type=int, help="Length of the fixed-length genotype")
parser.add_argument("--low", default=-10, type=float, help="Lower bound for the random initialization of the leaves")
parser.add_argument("--up", default=10, type=float, help="Upper bound for the random initialization of the leaves")
parser.add_argument("--types", default=None, type=str, help="This string must contain the range of constants for each variable in the format '#min_0,max_0,step_0,divisor_0;...;min_n,max_n,step_n,divisor_n'. All the numbers must be integers.")

parser.add_argument("--decay", default=0.99, type=float, help="The decay factor for the epsilon decay (eps_t = eps_0 * decay^t)")
parser.add_argument("--patience", default=50, type=int, help="Number of episodes to use as evaluation period for the early stopping")
parser.add_argument("--timeout", default=600, type=int, help="Maximum evaluation time, useful to continue the evolution in case of MemoryErrors")
parser.add_argument("--with_bias", action="store_true", help="if used, then the the condition will be (sum ...) < <const>, otherwise (sum ...) < 0")
parser.add_argument("--random_init", action="store_true", help="Randomly initializes the leaves in [-1, 1[")
parser.add_argument("--constant_range", default=1000, type=int, help="Max magnitude for the constants being used (multiplied *10^-3). Default: 1000 => constants in [-1, 1]")
parser.add_argument("--constant_step", default=1, type=int, help="Step used to generate the range of constants, mutliplied *10^-3")

parser.add_argument("--algorithm", default=0, type=int, help="grammatical_evolution or differential_evolution")
parser.add_argument("--grammar_angle", default=0, type=int, help="orthogonal or oblique")
parser.add_argument("--clients", default=1, type=int, help="number of flower clients")
parser.add_argument("--flsim", default=0, type=int, help="simulate flower framework by cloning population with best at each generation")

parser.add_argument("--s_day", default='1', help="start day")
parser.add_argument("--s_month", default='1', help="start month")
parser.add_argument("--s_year", default='1991', help="start year")
parser.add_argument("--e_day", default='1', help="end day")
parser.add_argument("--e_month", default='1', help="end month")
parser.add_argument("--e_year", default='1991', help="end year")

parser.add_argument("--rng", default=0, help="random location and date")


args = parser.parse_args()

lr = "auto" if args.learning_rate == "auto" else float(args.learning_rate)

#flwr legacy
Scalar = Union[bool, bytes, float, int, str]
Config = dict[str, Scalar]

class Code(Enum):
    """Client status codes."""
    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4

@dataclass
class Status:
    """Client status."""
    code: Code
    message: str

@dataclass
class Parameters:
    """Model parameters."""
    tensors: list[bytes]
    tensor_type: str

@dataclass
class GetParametersIns:
    """Parameters request for a client."""
    config: Config


@dataclass
class GetParametersRes:
    """Response when asked to return parameters."""

    status: Status
    parameters: Parameters

@dataclass
class FitIns:
    """Fit instructions for a client."""
    parameters: Parameters
    config: dict[str, Scalar]

@dataclass
class FitRes:
    """Fit response from a client."""
    status: Status
    parameters: Parameters
    num_examples: int
    metrics: dict[str, Scalar]


# Creation of an ad-hoc Leaf class
class CLeaf(RandomlyInitializedEpsGreedyLeaf):
    def __init__(self):
        super(CLeaf, self).__init__(args.n_actions, lr, args.df, args.eps, low=args.low, up=args.up)

# Creation of the EpsilonDecay Leaf
class EpsilonDecayLeaf(RandomlyInitializedEpsGreedyLeaf):
    """A eps-greedy leaf with epsilon decay."""

    def __init__(self):
        """
        Initializes the leaf
        """
        if not args.random_init:
            RandomlyInitializedEpsGreedyLeaf.__init__(
                self,
                n_actions=args.n_actions,
                learning_rate=lr,
                discount_factor=args.df,
                epsilon=args.eps,
                low=0,
                up=0
            )
        else:
            RandomlyInitializedEpsGreedyLeaf.__init__(
                self,
                n_actions=args.n_actions,
                learning_rate=lr,
                discount_factor=args.df,
                epsilon=args.eps,
                low=-1,
                up=1
            )

        self._decay = args.decay
        self._steps = 0

    def get_action(self):
        self.epsilon = self.epsilon * self._decay
        self._steps += 1
        return super().get_action()

def create_env(env_name, experiment_name, yearS=1991, monthS=1, dayS=1, yearE=1992, monthE=1, dayE=1):
    # Initialize Sinergym environment
    extra_params={'timesteps_per_hour' : 4,
              'runperiod' : (dayS,monthS,yearS,dayE,monthE,yearE)}
    env = gym.make(env_name, env_name=experiment_name, config_params=extra_params, max_ep_data_store_num=1)
    #eval_env = gym.make(env_name, env_name=experiment_name+'_EVALUATION')
    
    if not env.get_wrapper_attr('is_discrete'):
        # DISCRETIZATION
        # Defining new discrete space and action mapping function
        new_discrete_space = gym.spaces.Discrete(10) # Action values [0,9]
        def action_mapping_function(action):
            mapping = {
                0: [12, 30],
                1: [13, 30],
                2: [14, 29],
                3: [15, 28],
                4: [16, 28],
                5: [17, 27],
                6: [18, 26],
                7: [19, 25],
                8: [20, 24],
                9: [21, 23.25]
            }

            return mapping[action]
        # Apply the discretize wrapper
        env=DiscretizeEnv(env,discrete_space=new_discrete_space,action_mapping=action_mapping_function)
        #env = NormalizeObservation(env)
        #env = NormalizeAction(env)
    #env = LoggerWrapper(env)
    #env = CSVLogger(env)
    env = ReduceObservationWrapper(env, ['co2_emission'])
    #if not eval_env.get_wrapper_attr('is_discrete'):
    #    eval_env = NormalizeObservation(eval_env)
    #    eval_env = NormalizeAction(eval_env)
    #eval_env = LoggerWrapper(eval_env)
    #eval_env = CSVLogger(eval_env)
    return env#, eval_env

# Replacement function
def replace_in_strings(s, replacement_list):
    return re.sub(r"_in_(\d+)", lambda match: replacement_list[int(match.group(1))] if int(match.group(1)) < len(replacement_list) else match.group(0), s)
    
def program(program, _in_):
    program_fix = re.sub(r"_in_(\d+)", lambda match: f"_in_[{match.group(1)}]", program)
    local_vars = {"_in_": _in_}
    exec(program_fix, {}, local_vars)
    out = local_vars.get("out", None)
    return out

class SinergymClient():
    def __init__(self, seed, environment_name, episodes, episode_len, grammar_angle, input_space, types, constant_range, constant_step, with_bias, lambda_, generations, jobs, cxp, mp, mutation, crossover, genotype_len, selection, patience, flsim, s_day, s_month, s_year, e_day, e_month, e_year, random_init, n_actions, df, eps, decay, low, up, rng, timeout, env_seed):
        self.seed = seed
        self.environment_name = environment_name
        self.episodes = episodes
        self.episode_len = episode_len
        self.angle = grammar_angle
        self.input_space = input_space
        self.types = types
        self.constant_range = constant_range
        self.constant_step = constant_step
        self.with_bias = with_bias
        self.lambda_ = lambda_
        self.generations = generations
        self.jobs = jobs
        self.cxp = cxp
        self.mp = mp
        self.mutation = mutation
        self.crossover = crossover
        self.genotype_len = genotype_len
        self.selection = selection
        self.patience = patience
        self.flsim = flsim
        self.timeout = timeout
        
        self.s_day = s_day
        self.s_month = s_month
        self.s_year = s_year
        self.e_day = e_day
        self.e_month = e_month
        self.e_year = e_year
        
        self.random_init = random_init,
        self.n_actions = n_actions,
        self.df = df,
        self.eps = eps,
        self.decay = decay,
        self.low = low,
        self.up = up,
        
        self.env_seed = env_seed
        
        self.best_genotype = None
        self.server_genotype = None
        self.seed_offset = 0
        self.env_ins = ''
        self.best_phenotype = ''
        self.server_phenotype = ''

        self.best_phenotype_formatted = ''
        self.rng = rng
        
        self.env_names = ['Eplus-5zone-cool-discrete-stochastic-v1',
                            'Eplus-5zone-mixed-discrete-stochastic-v1',
                            'Eplus-5zone-hot-discrete-stochastic-v1',
                            'Eplus-5zoneAu-hot-discrete-stochastic-v1',
                            'Eplus-5zoneCo-hot-discrete-stochastic-v1',
                            'Eplus-5zoneCol-hot-discrete-stochastic-v1',
                            'Eplus-5zoneFi-cool-discrete-stochastic-v1',
                            'Eplus-5zoneIl-hot-discrete-stochastic-v1',
                            'Eplus-5zoneJpn-hot-discrete-stochastic-v1',
                            'Eplus-5zoneMdg-hot-discrete-stochastic-v1',
                            'Eplus-5zonePa-hot-discrete-stochastic-v1',
                            'Eplus-5zonePt-cool-discrete-stochastic-v1',
                            'Eplus-5zoneSp-hot-discrete-stochastic-v1',
                            'Eplus-5zoneSwe-hot-discrete-stochastic-v1']
        
        # Seeding of the random number generators
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.experiment_name = f"SB3_PPO-{environment_name[0]}_{datetime.today().strftime('%Y-%m-%d_%H:%M')}"
        
        if self.angle == 0:
            self.orthogonal_grammar()
            self.node = CLeaf
        elif self.angle == 1:
            self.oblique_grammar()
            self.node = EpsilonDecayLeaf

    def orthogonal_grammar(self):
        self.grammar = {
            "bt": ["<if>"],
            "if": ["if <condition>:{<action>}else:{<action>}"],
            "condition": [f"_in_{k}<comp_op><const_type_{k}>" for k in range(self.input_space)],
            "action": ["out=_leaf;leaf=\"_leaf\"", "<if>"],
            "comp_op": [" < ", " > "],
        }
        types = self.types if self.types is not None else ";".join(["0,10,1,10" for _ in range(self.input_space)])
        types = types.replace("#", "")
        assert len(types.split(";")) == self.input_space, f"Expected {self.input_space} types, got {len(types.split(';'))}."

        for index, type_ in enumerate(types.split(";")):
            rng = type_.split(",")
            start, stop, step, divisor = map(int, rng)
            consts_ = list(map(str, [float(c) / divisor for c in range(start, stop, step)]))
            self.grammar[f"const_type_{index}"] = consts_

        #print(self.grammar)

    def oblique_grammar(self):
        types = self.types if self.types is not None else ";".join(["0,10,1,10" for _ in range(self.input_space)])
        types = types.replace("#", "")
        assert len(types.split(";")) == self.input_space, f"Expected {self.input_space} types, got {len(types.split(';'))}."
        
        consts = {}
        for index, type_ in enumerate(types.split(";")):
            rng = type_.split(",")
            start, stop, step, divisor = map(int, rng)
            consts_ = list(map(str, [float(c) / divisor for c in range(start, stop, step)]))
            consts[index] = (consts_[0], consts_[-1])
        
        oblique_split = "+".join([f"<const> * (_in_{i} - {consts[i][0]})/({consts[i][1]} - {consts[i][0]})" for i in range(self.input_space)])
        
        self.grammar = {
            "bt": ["<if>"],
            "if": ["if <condition>:{<action>}else:{<action>}"],
            "action": ["out=_leaf;leaf=\"_leaf\"", "<if>"],
            # "const": ["0", "<nz_const>"],
            "const": [str(k/1000) for k in range(-self.constant_range,self.constant_range+1,self.constant_step)]
        }
        
        if not self.with_bias:
            self.grammar["condition"] = [oblique_split + " < 0"]
        else:
            self.grammar["condition"] = [oblique_split + " < <const>"]
        
        #print(self.grammar)

    def get_parameters(self, ins):
        # Return the tree parameters
        genotype_bytes = pickle.dumps(self.best_genotype)
        parameters = Parameters(tensors=[genotype_bytes], tensor_type="DEAP_Individual")
        return GetParametersRes(
            status=Status(code=Code.OK, message="Parameters retrieved successfully"),
            parameters=parameters
        )

    def set_parameters(self, parameters):
        genotype_bytes = parameters.tensors[0]
        received_genotype = pickle.loads(genotype_bytes)
        self.server_genotype = received_genotype
        self.best_genotype = received_genotype

    def fit_fcn(self, x):
        return self.train_GE(self.node, x[1], x[2])

    def fit(self):
        #temp env to get obs vars, had problems with self.env_ins
        env_t = gym.make(self.environment_name[0], env_name=self.experiment_name)
        env_t = ReduceObservationWrapper(env_t, ['co2_emission'])
        self.env_ins = env_t.get_wrapper_attr('observation_variables')
        env_t.close()
        
        previous = []
        if self.best_genotype is not None:
            previous.append(self.best_genotype)
            print('==== Previous best genotype used for training ====')
        
        self.seed_offset = self.seed_offset + 1
        with parallel_backend("multiprocessing"):
            pop, log, hof, best_leaves = grammatical_evolution(
                self.fit_fcn,
                inputs=self.input_space,
                leaf=self.node,
                individuals=self.lambda_,
                generations=self.generations,
                jobs=self.jobs,
                cx_prob=self.cxp,
                m_prob=self.mp,
                logfile=logfile,
                seed=self.seed+self.seed_offset,
                mutation=self.mutation,
                crossover=self.crossover,
                initial_len=self.genotype_len,
                selection=self.selection,
                timeout=self.timeout,
                pops=previous,#[self.server_genotype,self.best_genotype],
                flsim=(self.flsim!=0))
        
        for hofn in hof:
            phenotype, _ = GrammaticalEvolutionTranslator(self.grammar).genotype_to_str(hofn)
            phenotype = phenotype.replace('leaf="_leaf"', '')

            for k in range(50000):  # Iterate over all possible leaves
                key = f"leaf_{k}"
                if key in best_leaves:
                    v = best_leaves[key].q
                    phenotype = phenotype.replace("out=_leaf", f"out={np.argmax(v)}", 1)
                else:
                    break

            #print(str(log) + "\n")
            print(str(hofn) + "\n")
            _in_pos = [m.start() for m in re.finditer('_in_', phenotype)]
            phenotype_formatted = replace_in_strings(phenotype, self.env_ins)
            #print(phenotype_formatted + "\n")
            #print(f"best_fitness: {hofn.fitness.values[0]}")
            
            eval_fit = self.evaluate(phenotype)
            eval_fit = np.mean(eval_fit)
            print(f'Evals: {eval_fit}')
        
            if self.best_genotype is None or hofn.fitness.values[0] > self.best_genotype.fitness.values[0]:
                self.best_genotype = hofn
                self.best_phenotype = phenotype
                self.best_phenotype_formatted = phenotype_formatted
        
        num_evaluation_episodes = self.episodes
        
        #REMOVE EPLUS LOGS ============= to avoid memory usage creep ==
        import os
        import shutil
        import glob
        
        log_dir = os.path.dirname(os.path.realpath(__file__))
        simulation_name = "Eplus-env-SB3_PPO"
        pattern = os.path.join(log_dir, simulation_name + "*")
        folders = glob.glob(pattern)
        
        if folders:
            for folder_n in folders:
                folder_to_delete = folder_n
                #print(f"Deleting folder: {folder_to_delete}")
                shutil.rmtree(folder_to_delete)
        #else:
            #print("No matching folder found.")
        #=============================================================
        
        return [self.best_genotype.fitness.values[0], eval_fit]

    def evaluate(self, phenotype):
        #self.set_parameters(ins.parameters)
        
        #if self.best_phenotype == '':
        #    if self.server_genotype is not None:
        #        phenotype, _ = GrammaticalEvolutionTranslator(self.grammar).genotype_to_str(self.server_genotype)
        #        print('==== Server genotype used for evaluation ====')
        #    elif self.best_genotype is not None:
        #        phenotype, _ = GrammaticalEvolutionTranslator(self.grammar).genotype_to_str(self.best_genotype)
        #        print('==== Local genotype used for evaluation ====')
        #    else:
        #        phenotype = ''
        #        print('==== Nothing used for evaluation ====')
        #else:
        #    phenotype = self.best_phenotype
        #    print('==== Local phenotype used for evaluation ====')
        
        cum_fit_scores = []
        for e_name in self.env_names:
            random.seed(self.seed)
            np.random.seed(self.seed)
            global_cumulative_rewards = []
            
            #fixed env, alternating env, random env
            #env_index = 0
            env_index = 1%2
            #env_index = random.randint(0, len(self.env_names))
            
            dayS = int(self.s_day[env_index % len(self.s_day)])
            monthS = int(self.s_month[env_index % len(self.s_month)])
            yearS = int(self.s_year[env_index % len(self.s_year)])
            dayE = int(self.e_day[env_index % len(self.e_day)])
            monthE = int(self.e_month[env_index % len(self.e_month)])
            yearE = int(self.e_year[env_index % len(self.e_year)])
            
            
            if self.rng:
                import time
                random.seed(int(time.time()))
                yearS = yearE = random.randint(1991, 1992)
                monthS = monthE = random.randint(1, 12)
                dayS = dayE = random.randint(1, 21)
                dayE = dayE+7
                e = create_env(e_name, self.experiment_name, yearS, monthS, dayS, yearE, monthE, dayE)
                print(f"=====BDT tested on: {e_name}, {dayS}/{monthS}/{yearS}-{dayE}/{monthE}/{yearE}=====")
            else:
                e = create_env(e_name, self.experiment_name, yearS, monthS, dayS, yearE, monthE, dayE)
                print(f"=====BDT tested on: {e_name}, {dayS}/{monthS}/{yearS}-{dayE}/{monthE}/{yearE}=====")
            self.env_ins = e.get_wrapper_attr('observation_variables')
            initial_perf = None
            
            try:
                for iteration in range(self.episodes):
                    obs, _ = e.reset(seed=iteration)
                    cum_rew = 0
                    action = 0
                    previous = None
                    
                    for t in range(self.episode_len):
                        if self.angle == 1:
                            obs = list(obs.flatten())
                        input_ = []
                        input_.extend(obs)
                        
                        action = program(phenotype, input_)

                        previous = obs[:]

                        obs, rew, done, trunc, info = e.step(action)

                        cum_rew += rew

                        if done:
                            break
                        
                    global_cumulative_rewards.append(cum_rew)
                    
                    if self.angle == 1:
                        # Check stopping criterion

                        if initial_perf is None and iteration >= self.patience:
                            initial_perf = np.mean(global_cumulative_rewards)
                        elif iteration % self.patience == 0 and iteration > self.patience:
                            if np.mean(global_cumulative_rewards[-self.patience:]) - initial_perf < 0:
                                break
                            initial_perf = np.mean(global_cumulative_rewards[-self.patience:])
            except KeyError as ex:
                if len(global_cumulative_rewards) == 0:
                    global_cumulative_rewards = [-3000000]
            except Exception as ex:
                print(f"Exception: {ex}")
                traceback.print_exc()
                if len(global_cumulative_rewards) == 0:
                    global_cumulative_rewards = [-3000000]
            
            e.close()
            
            
            if self.angle == 0:
                fitness = np.mean(global_cumulative_rewards),
                std_sc = np.std(global_cumulative_rewards)
            elif self.angle == 1:
                fitness = np.mean(global_cumulative_rewards[-self.patience:]),
                std_sc = np.std(global_cumulative_rewards[-self.patience:])
            
            fitness_score = fitness
            print(f"==== {e_name} Eval fitness: {fitness_score[0]}, std: {std_sc} ====")
            num_evaluation_episodes = self.episodes
            
            cum_fit_scores.append(fitness_score[0])
            
        return cum_fit_scores
        
    def train_GE(self, leaf, genotype, gen=0):
        if self.angle == 1:
            repeatable_random_seed = sum(genotype) % (2 ** 31)
            random.seed(self.seed + repeatable_random_seed)
            np.random.seed(self.seed + repeatable_random_seed)
        phenotype, _ = GrammaticalEvolutionTranslator(self.grammar).genotype_to_str(genotype)
        bt = PythonDT(phenotype, leaf)
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        global_cumulative_rewards = []
        
        #fixed env, alternating env, random env
        #env_index = 0
        env_index = gen%2%len(self.environment_name)
        #env_index = random.randint(0, len(self.env_names))
        
        dayS = int(self.s_day[env_index % len(self.s_day)])
        monthS = int(self.s_month[env_index % len(self.s_month)])
        yearS = int(self.s_year[env_index % len(self.s_year)])
        dayE = int(self.e_day[env_index % len(self.e_day)])
        monthE = int(self.e_month[env_index % len(self.e_month)])
        yearE = int(self.e_year[env_index % len(self.e_year)])
        
        
        if self.rng:
            import time
            random.seed(int(time.time()))
            yearS = yearE = random.randint(1991, 1992)
            monthS = monthE = random.randint(1, 12)
            dayS = dayE = random.randint(1, 21)
            dayE = dayE+7
            e = create_env(self.env_names[random.randint(0, len(self.env_names)-1)], self.experiment_name, yearS, monthS, dayS, yearE, monthE, dayE)
        else:
            e = create_env(self.environment_name[env_index], self.experiment_name, yearS, monthS, dayS, yearE, monthE, dayE)
        self.env_ins = e.get_wrapper_attr('observation_variables')
        initial_perf = None
        
        try:
            for iteration in range(self.episodes):
                obs, _ = e.reset(seed=iteration)
                bt.new_episode()
                cum_rew = 0
                action = 0
                previous = None

                for t in range(self.episode_len):
                    if self.angle == 1:
                        obs = list(obs.flatten())
                    
                    action = bt(obs)

                    previous = obs[:]

                    obs, rew, done, trunc, info = e.step(action)

                    # e.render()
                    bt.set_reward(rew)

                    cum_rew += rew

                    if done:
                        break
                        
                if self.angle == 0:
                    bt.set_reward(rew)

                bt(obs)
                global_cumulative_rewards.append(cum_rew)
                
                if self.angle == 1:
                    # Check stopping criterion
                    if initial_perf is None and iteration >= self.patience:
                        initial_perf = np.mean(global_cumulative_rewards)
                    elif iteration % self.patience == 0 and iteration > self.patience:
                        if np.mean(global_cumulative_rewards[-self.patience:]) - initial_perf < 0:
                            break
                        initial_perf = np.mean(global_cumulative_rewards[-self.patience:])
        except KeyError as ex:
            #print("Invalid tree")
            #traceback.print_exc()
            if len(global_cumulative_rewards) == 0:
                global_cumulative_rewards = [-3000000]
        except Exception as ex:
            print(f"Exception: {ex}")
            traceback.print_exc()
            if len(global_cumulative_rewards) == 0:
                global_cumulative_rewards = [-3000000]
        
        e.close()
        
        
        if self.angle == 0:
            fitness = np.mean(global_cumulative_rewards),
        elif self.angle == 1:
            fitness = np.mean(global_cumulative_rewards[-self.patience:]),
        
        return fitness, bt.leaves

if __name__ == "__main__":
    
    # Setup of the logging
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"logs/gym/{date}_{''.join(np.random.choice(list(string.ascii_lowercase), size=8))}"
    logfile = os.path.join(logdir, "log.txt")
    os.makedirs(logdir)

    # Log all the parameters
    with open(logfile, "a") as f:
        vars_ = locals().copy()
        for k, v in vars_.items():
            f.write(f"{k}: {v}\n")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    env_seed = random.randint(0, 13)
    
    client =  SinergymClient(
        environment_name=args.environment_name.split('§'),#['Eplus-5zone-hot-discrete-stochastic-v1','Eplus-5zoneMdg-hot-discrete-stochastic-v1'],
        seed=args.seed,
        episodes=args.episodes,
        episode_len=args.episode_len,
        grammar_angle=args.grammar_angle,
        input_space=args.input_space,
        types=args.types,
        constant_range=args.constant_range,
        constant_step=args.constant_step,
        with_bias=args.with_bias,
        lambda_=args.lambda_,
        generations=args.generations,
        jobs=args.jobs,
        cxp=args.cxp,
        mp=args.mp,
        mutation=args.mutation,
        crossover=args.crossover,
        genotype_len=args.genotype_len,
        selection=args.selection,
        patience=args.patience,
        timeout=args.timeout,
        flsim=args.flsim,
        
        s_day=args.s_day.split('§'),
        s_month=args.s_month.split('§'),
        s_year=args.s_year.split('§'),
        e_day=args.e_day.split('§'),
        e_month=args.e_month.split('§'),
        e_year=args.e_year.split('§'),
        
        random_init=args.random_init,
        n_actions=args.n_actions,
        df=args.df,
        eps=args.eps,
        decay=args.decay,
        low=args.low,
        up=args.up,
        env_seed=env_seed,
        rng=args.rng,
        )
    
    
    # Start Flower client
    results_fit = client.fit()
    
    prog="""
if _in_13 > 49.0:
    out=1

else:
    out=8
    """
    #client.evaluate(prog)
    
    # LOG RESULTS
    resdir = f"data/{date}_{''.join(np.random.choice(list(string.ascii_lowercase), size=8))}"
    resfile = os.path.join(resdir, "res.txt")
    genfile = os.path.join(resdir, "gen.txt")
    if not os.path.exists(resdir):
        os.makedirs(resdir)
    with open(resfile, "a") as f:
        f.write(f"top best:{client.best_genotype.fitness.values[0]}; genotype:{client.best_genotype}\n")
    with open(genfile, "a") as f:
        f.write(f"run genotype:{client.best_phenotype_formatted}\n")
