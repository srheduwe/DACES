from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
import time
import sys
from omegaconf import OmegaConf
import numpy as np
from AE_gen import ae_gen

np.random.seed(seed=42)

class EA:
    def __init__(self, configuration):
        self.configuration = configuration

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        # stdev_init = Float("stdev_init", (0.05, 0.1), default=0.075)
        stdev_init = Float("stdev_init", (0.005, 0.05), default=0.01)
        # stdev_max = Float("stdev_max", (2.5, 3.5), default=3.0)
        stdev_max = Float("stdev_max", (1.0, 3.5), default=2.0)
        e_sigma = Float("e_sigma", (0.1, 0.5), default=0.3)
        c_m = Float("c_m", (0.25, 0.75), default=0.5)
        c_sigma = Float("c_sigma", (0.005, 0.01), default=0.0075)
        # scaling_factor = Integer("scaling_factor", (3, 19), default=15) 
        # popsize = Integer("popsize", (20, 100), default=60) 
        popsize = Integer("popsize", (15, 80), default=40) 
        c_sigma_ratio = Float("c_sigma_ratio", (1.5, 2.0), default=1.75)
        damp_sigma = Float("damp_sigma", (1.0, 1.01), default=1.005)
        damp_sigma_ratio = Float("damp_sigma_ratio", (0.5, 1.0), default=0.75)
        c_c = Float("c_c", (0.01, 0.1), default=0.05)
        c_c_ratio = Float("c_c_ratio", (1.0, 1.5), default=1.25)
        # c_1 = Float("c_1", (0.0005, 0.001), default=0.0007)
        c_1 = Float("c_1", (0.0005, 0.002), default=0.001)
        # c_1_ratio = Float("c_1_ratio", (1.25, 2.5), default=1.75)
        c_1_ratio = Float("c_1_ratio", (1.5, 3.5), default=2.5)
        c_mu = Float("c_mu", (0.001, 0.01), default=0.005) 
        c_mu_ratio = Float("c_mu_ratio", (1.5, 3.0), default=2.25) 

        cs.add([stdev_init, stdev_max, 
                e_sigma, c_m, c_sigma, #scaling_factor,
                c_sigma_ratio, damp_sigma, damp_sigma_ratio, c_c, c_c_ratio, c_1, c_1_ratio, c_mu, c_mu_ratio, 
                popsize,
                ])
        return cs

    def train(self, config: Configuration, seed: int = 0, budget: int = 1000) -> float:
        configuration = self.configuration
        
        configuration.parameters.stdev_init = config.get("stdev_init")
        configuration.parameters.stdev_max = config.get("stdev_max")
        configuration.parameters.e_sigma = config.get("e_sigma")
        configuration.parameters.c_m = config.get("c_m")
        configuration.parameters.c_sigma = config.get("c_sigma")
        # configuration.parameters.scaling_factor = config.get("scaling_factor")
        configuration.parameters.popsize = config.get("popsize")
        configuration.parameters.c_sigma_ratio = config.get("c_sigma_ratio")
        configuration.parameters.damp_sigma = config.get("damp_sigma")
        configuration.parameters.damp_sigma_ratio = config.get("damp_sigma_ratio")
        configuration.parameters.c_c = config.get("c_c")
        configuration.parameters.c_c_ratio = config.get("c_c_ratio")
        configuration.parameters.c_1 = config.get("c_1")
        configuration.parameters.c_1_ratio = config.get("c_1_ratio")
        configuration.parameters.c_mu = config.get("c_mu")
        configuration.parameters.c_mu_ratio = config.get("c_mu_ratio")

        print(configuration.parameters)
        mean_eval_all = []
        for seed in [0,1,2,3,4,5,6,7,8,9]:
            for model in configuration.experiment.models:
                print("Seed: ", seed, ", model: ", model)
                configuration.experiment.model = model
                configuration.experiment.seed_images = seed
                mean_eval = ae_gen(configuration, np.ceil(budget))
                mean_eval_all += [mean_eval]
                overall_mean = np.mean(mean_eval_all)
                print(overall_mean)
                if overall_mean > 600*1000*1: # 1 only for bilinear interpolation and NNI, else 5
                    return overall_mean
        
        return overall_mean

if __name__ == "__main__":
    start = time.process_time()
    experiment = str(sys.argv[1])
    configuration = OmegaConf.load(experiment)
    ea = EA(configuration=configuration)

    facades: list[AbstractFacade] = []
    for intensifier_object in [Hyperband]: #SuccessiveHalving, 
        # Define our environment variables
        scenario = Scenario(
            ea.configspace,
            name="Cifar100",
            deterministic=True,
            use_default_config=True,
            walltime_limit=20*24*60*60,  # After XX seconds, we stop the hyperparameter optimization
            n_trials=100,  # Evaluate max XX different trials
            min_budget=600, 
            max_budget=30000, 
            n_workers=1,
        )
        
        # We want to run five random configurations before starting the optimization.
        initial_design = MFFacade.get_initial_design(scenario, n_configs=30)
        
        # Create our intensifier
        intensifier = intensifier_object(scenario, incumbent_selection="highest_budget", eta=7)
        
        # Create our SMAC object and pass the scenario and the train method
        smac = MFFacade(
            scenario,
            ea.train,
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=False,
        ) #th, name, eta, min_budget
        
        # Let's optimize
        incumbent = smac.optimize()
        
        # Let's calculate the cost of the incumbent
        incumbent_cost = smac.validate(incumbent)
        print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")

        facades.append(smac)

    # Let's plot it
    print("incumbent_cost: ", incumbent_cost)
    print("incumbent.config_space ", incumbent.config_space)
    print("incumbent.values() ", incumbent.values())
    end = time.process_time()
    print(end-start)