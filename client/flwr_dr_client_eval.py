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
parser.add_argument("--n_actions", default=10, type=int, help="The number of action that the agent can perform in the environment")
parser.add_argument("--learning_rate", default="auto", help="The learning rate to be used for Q-learning. Default is: 'auto' (1/k)")
parser.add_argument("--df", default=0.9, type=float, help="The discount factor used for Q-learning")
parser.add_argument("--eps", default=0.05, type=float, help="Epsilon parameter for the epsilon greedy Q-learning")
parser.add_argument("--input_space", default=16, type=int, help="Number of inputs given to the agent")
parser.add_argument("--episodes", default=5, type=int, help="Number of episodes that the agent faces in the fitness evaluation phase")
parser.add_argument("--episode_len", default=2976, type=int, help="The max length of an episode in timesteps")
parser.add_argument("--lambda_", default=1, type=int, help="Population size")
parser.add_argument("--generations", default=1, type=int, help="Number of generations")
parser.add_argument("--cxp", default=0.3, type=float, help="Crossover probability")
parser.add_argument("--mp", default=1, type=float, help="Mutation probability")
parser.add_argument("--mutation", default="function-tools.mutUniformInt#low-0#up-40000#indpb-0.1", type=string_to_dict, help="Mutation operator. String in the format function-value#function_param_-value_1... The operators from the DEAP library can be used by setting the function to 'function-tools.<operator_name>'. Default: Uniform Int Mutation")
parser.add_argument("--crossover", default="function-tools.cxOnePoint", type=string_to_dict, help="Crossover operator, see Mutation operator. Default: One point")
parser.add_argument("--selection", default="function-tools.selTournament#tournsize-2", type=string_to_dict, help="Selection operator, see Mutation operator. Default: tournament of size 2")

parser.add_argument("--genotype_len", default=100, type=int, help="Length of the fixed-length genotype")
parser.add_argument("--low", default=-1, type=float, help="Lower bound for the random initialization of the leaves")
parser.add_argument("--up", default=1, type=float, help="Upper bound for the random initialization of the leaves")
parser.add_argument("--types", default='#1,13,1,1;1,31,1,1;0,23,1,1;-10,50,1,1;0,100,5,1;0,25,1,1;0,360,45,1;0,100,10,1;0,1000,100,1;-10,50,1,1;-10,50,1,1;-10,50,1,1;0,100,1,1;0,50,1,1;0,300,10,1000;0,300,10,1000', type=str, help="This string must contain the range of constants for each variable in the format '#min_0,max_0,step_0,divisor_0;...;min_n,max_n,step_n,divisor_n'. All the numbers must be integers.")

parser.add_argument("--decay", default=0.99, type=float, help="The decay factor for the epsilon decay (eps_t = eps_0 * decay^t)")
parser.add_argument("--patience", default=30, type=int, help="Number of episodes to use as evaluation period for the early stopping")
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
parser.add_argument("--s_year", default='1990', help="start year")
parser.add_argument("--e_day", default='7', help="end day")
parser.add_argument("--e_month", default='2', help="end month")
parser.add_argument("--e_year", default='1990', help="end year")

parser.add_argument("--place", default='ny', help="weather file location")


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

def program(input_, place):
    #ref list:
    #0 month
    #1 day_of_month
    #2 hour
    #3 outdoor_temperature
    #4 outdoor_humidity
    #5 wind_speed
    #6 wind_direction
    #7 diffuse_solar_radiation
    #8 direct_solar_radiation
    #9 htg_setpoint
    #10 clg_setpoint
    #11 air_temp
    #12 air_hum
    #13 occup
    #--14 co2
    #14 hvac_dem
    #15 tot_hvac
    
    # Insert program here
    if place == 'human':
        if input_[11] < 25:
            out=8
        else:
            out=4
    elif place == "fi":
        if -0.419 * (input_[0] - 1.0)/(12.0 - 1.0)+0.297 * (input_[1] - 0.0)/(23.0 - 0.0)+-0.156 * (input_[2] - -10.0)/(49.0 - -10.0)+-0.777 * (input_[3] - 0.0)/(99.0 - 0.0)+-0.431 * (input_[4] - 0.0)/(299.0 - 0.0)+0.855 * (input_[5] - 0.0)/(999.0 - 0.0)+-0.474 * (input_[6] - -10.0)/(49.0 - -10.0)+0.615 * (input_[7] - -10.0)/(49.0 - -10.0)+-0.347 * (input_[8] - -10.0)/(49.0 - -10.0)+0.569 * (input_[9] - 0.0)/(99.0 - 0.0)+-0.385 * (input_[10] - 0.0)/(49.0 - 0.0) < 0:
            if -0.159 * (input_[0] - 1.0)/(12.0 - 1.0)+-0.579 * (input_[1] - 0.0)/(23.0 - 0.0)+-0.839 * (input_[2] - -10.0)/(49.0 - -10.0)+0.798 * (input_[3] - 0.0)/(99.0 - 0.0)+-0.744 * (input_[4] - 0.0)/(299.0 - 0.0)+0.824 * (input_[5] - 0.0)/(999.0 - 0.0)+0.565 * (input_[6] - -10.0)/(49.0 - -10.0)+0.874 * (input_[7] - -10.0)/(49.0 - -10.0)+0.76 * (input_[8] - -10.0)/(49.0 - -10.0)+-0.421 * (input_[9] - 0.0)/(99.0 - 0.0)+-0.204 * (input_[10] - 0.0)/(49.0 - 0.0) < 0:
                if -0.755 * (input_[0] - 1.0)/(12.0 - 1.0)+0.35 * (input_[1] - 0.0)/(23.0 - 0.0)+-0.359 * (input_[2] - -10.0)/(49.0 - -10.0)+-0.589 * (input_[3] - 0.0)/(99.0 - 0.0)+0.389 * (input_[4] - 0.0)/(299.0 - 0.0)+0.257 * (input_[5] - 0.0)/(999.0 - 0.0)+-0.336 * (input_[6] - -10.0)/(49.0 - -10.0)+-0.611 * (input_[7] - -10.0)/(49.0 - -10.0)+-0.57 * (input_[8] - -10.0)/(49.0 - -10.0)+0.963 * (input_[9] - 0.0)/(99.0 - 0.0)+-0.484 * (input_[10] - 0.0)/(49.0 - 0.0) < 0:
                    out=4
                else:
                    out=0
            else:
                if -0.443 * (input_[0] - 1.0)/(12.0 - 1.0)+-0.201 * (input_[1] - 0.0)/(23.0 - 0.0)+-0.762 * (input_[2] - -10.0)/(49.0 - -10.0)+0.463 * (input_[3] - 0.0)/(99.0 - 0.0)+-0.134 * (input_[4] - 0.0)/(299.0 - 0.0)+0.219 * (input_[5] - 0.0)/(999.0 - 0.0)+-0.016 * (input_[6] - -10.0)/(49.0 - -10.0)+0.496 * (input_[7] - -10.0)/(49.0 - -10.0)+0.604 * (input_[8] - -10.0)/(49.0 - -10.0)+-0.5 * (input_[9] - 0.0)/(99.0 - 0.0)+0.977 * (input_[10] - 0.0)/(49.0 - 0.0) < 0:
                    out=4
                else:
                    out=9
        else:
            out=0
    
    # End program
    return out

class SinergymClient():
    def __init__(self, seed, environment_name, episodes, episode_len, grammar_angle, input_space, types, constant_range, constant_step, with_bias, lambda_, generations, jobs, cxp, mp, mutation, crossover, genotype_len, selection, patience, flsim, s_day, s_month, s_year, e_day, e_month, e_year, random_init, n_actions, df, eps, decay, low, up, place):
        self.seed = seed
        self.seed_inc = 0
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
        
        self.best_genotype = None
        self.server_genotype = None
        self.seed_offset = 0
        self.env_ins = ''

        self.best_phenotype_formatted = ''
        
        self.place=place
        
        # Seeding of the random number generators
        random.seed(self.seed+self.seed_inc)
        np.random.seed(self.seed+self.seed_inc)
        
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
    
    def evaluate_GE_min(self, s):
        if self.angle == 1:
            random.seed(self.seed + s)
            np.random.seed(self.seed + s)
        
        # Seeding of the random number generators
        self.seed_inc = self.seed_inc+1
        random.seed(self.seed+self.seed_inc)
        np.random.seed(self.seed+self.seed_inc)
        global_cumulative_rewards = []
        
        #fixed enc, alternating env, random env
        env_index = 0
        #env_index = gen%2
        #env_index = random.randint(0, len(self.environment_name))
        
        dayS = int(self.s_day[env_index % len(self.s_day)])
        monthS = int(self.s_month[env_index % len(self.s_month)])
        yearS = int(self.s_year[env_index % len(self.s_year)])
        dayE = int(self.e_day[env_index % len(self.e_day)])
        monthE = int(self.e_month[env_index % len(self.e_month)])
        yearE = int(self.e_year[env_index % len(self.e_year)])
        
        #yearS = yearE = random.randint(1991, 1992)
        monthS = monthE = random.randint(1, 12)
        #dayS = dayE = random.randint(1, 25)
        #dayE = dayE+3
        
        #with HiddenPrints():
        
        #e = create_env(self.environment_name[env_index], self.experiment_name, yearS, monthS, dayS, yearE, monthE, dayE)
        e = create_env(self.environment_name[env_index], self.experiment_name, 1991, 1, 1, 1991, 2, 1)
        self.env_ins = e.get_wrapper_attr('observation_variables')
        #print(f"env_ins:{self.env_ins};")
        initial_perf = None
        
        try:
            for iteration in range(self.episodes):
                obs, _ = e.reset(seed=iteration+self.seed)
                cum_rew = 0
                action = 0
                previous = None

                for t in range(self.episode_len):
                    if self.angle == 1:
                        obs = list(obs.flatten())
                    
                    input_ = []
                    input_.extend(obs)
                    action = program(input_, self.place)

                    previous = obs[:]

                    obs, rew, done, trunc, info = e.step(action)

                    # e.render()

                    cum_rew += rew

                    if done:
                        break
                    
                global_cumulative_rewards.append(cum_rew)
                #print(f"====== Iteration {iteration+1} Reward: {cum_rew} ======")
                
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
        
        print(f"Fit: {fitness}")
        return fitness

if __name__ == "__main__":
    env_names = ['Eplus-5zone-cool-discrete-stochastic-v1','Eplus-5zone-mixed-discrete-stochastic-v1','Eplus-5zone-hot-discrete-stochastic-v1','Eplus-5zoneAu-hot-discrete-stochastic-v1','Eplus-5zoneCo-hot-discrete-stochastic-v1','Eplus-5zoneCol-hot-discrete-stochastic-v1','Eplus-5zoneFi-cool-discrete-stochastic-v1','Eplus-5zoneIl-hot-discrete-stochastic-v1','Eplus-5zoneJpn-hot-discrete-stochastic-v1','Eplus-5zoneMdg-hot-discrete-stochastic-v1','Eplus-5zonePa-hot-discrete-stochastic-v1','Eplus-5zonePt-cool-discrete-stochastic-v1','Eplus-5zoneSp-hot-discrete-stochastic-v1','Eplus-5zoneSwe-hot-discrete-stochastic-v1']
    #env_names = ['Eplus-5zone-mixed-discrete-stochastic-v1']
    
    n_runs=10
    
    for i,name in enumerate(env_names):
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
        
        client =  SinergymClient(
            environment_name=[name],
            seed=i,
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
            
            place=args.place
            )
        
        genotype_bytes = pickle.dumps(client.best_genotype)
        parameters = Parameters(tensors=[genotype_bytes], tensor_type="DEAP_Individual")
        #empty_parameters = Parameters(tensors=[], tensor_type="")
        empty_fitins = FitIns(parameters=parameters, config={})
        
        print(f"=====BDT tested on: {name}=====")
        scores = [client.evaluate_GE_min(s) for s in range(i, i + n_runs)]
        print(f"Mean: {np.mean(scores)} ({np.mean(scores)/31}/day); Std: {np.std(scores)} ({np.std(scores)/31}/day)")
