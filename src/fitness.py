import torch 
from torch import linalg as LA
from evotorch import Problem, SolutionBatch
import time

class fitness(Problem):
    def __init__(self, 
                 d: int,                    # Dimensionality of the problem
                 label: int,                # Label of the original image
                 I: torch.Tensor,           # Original image
                 dim: int,                  # Length/Width of original image (for ImageNet=224)  
                 height: int,               # Height of lower image subspace
                 width: int,                # Width of lower image subspace
                 upsizer: list,             # upsizer for upscaling image subspace
                 net,                       # Neural Network for inference
                 exp_update,                # Whether e_sigma is used, defaults to None
                 e_sigma: float,            # Step size enlarger of dynamic configuration
                 device: bool,              
                 start_time: float,
                 query_budget: int,         # Query budget, most often 30.000
                 abl_c: bool,               # Whether ablation should be done on the dynamic configuration strategy
                 norm):                     # Norm to be used, most often 2 (euclidian norm)
        super().__init__(
            objective_sense="min",
            solution_length=d,
            bounds=(torch.zeros(height*width*3), torch.ones(height*width*3)),
            device=device,
            )

        self._label = label
        self._I = I                 
        self._dim = dim
        self._height = height
        self._width = width
        self._upsizer = upsizer           
        self._net = net
        self._exp_update = exp_update
        self._e_sigma = e_sigma
        self._device = device
        self._query_counter = torch.tensor([], device=device)
        self._best_norm = torch.tensor([], device=device)
        self._time = torch.tensor([], device=device)
        self._label_pun = 1000
        self._C_ben = False                 # indicates, whether any instance has been classified as benign before
        self._start_time = start_time
        self._query_budget = query_budget
        self._abl_c = abl_c
        self._best_Instance = None
        self._norm = norm
    
    def _evaluate_batch(self, solution: SolutionBatch):
        perturbation = solution.values                                  # The current population=perturbation
        popsize = perturbation.shape[0]
        upsizer = self._upsizer
        height, width = self._height, self._width
        pun = self._label_pun
        dim = self._dim
        
        perturbation = (perturbation.clone().reshape(popsize, 3, self._height, self._width)).float() 
        perturbed_image = self._I.unsqueeze(0).repeat(popsize, 1, 1, 1).float() # create tensor with popsize times the reference image

        if isinstance(upsizer, list):                                   # This is the case when using subspace activation and grid downsizing
            upsizer[0] = torch.arange(popsize).reshape(popsize, 1, 1, 1)# Update the popsize in the upsizer (only necessary if popsize is not given before)
            perturbed_image[upsizer] += perturbation                    # Add perturbation to original image
        elif upsizer:                                                   # For nearest neighbour and bilinear interpolation
            perturbation = upsizer(perturbation)                        # Scale perturbation to original image size
            perturbed_image += perturbation                             # Add perturbation to original image
            height = self._I.shape[1]
            width = self._I.shape[2]
        else: 
            perturbed_image += perturbation                             # Only for small images like CIFAR, when we search in the orginal image space

        perturbed_image = torch.clamp(perturbed_image, min=0.0, max=1.0)
        norms = LA.vector_norm(perturbation.reshape(popsize, 3*height*width), ord=float(self._norm), axis=1)
        f = norms.detach().clone()
        queries = torch.zeros(1, device=self._device)

        if not self._C_ben:                                     # Stage 1 of query strategy
            queried_instances = norms == norms.min()                    # We only query the nearest image
            no_of_queried_instances = queried_instances.sum()
            
            preds = self._net(perturbed_image[queried_instances].reshape(no_of_queried_instances, 3, dim, dim)).argmax(1)
            label_puns = torch.where(preds == self._label, pun, 0)            
            queries += no_of_queried_instances.unsqueeze(0)

            if label_puns.sum() == pun*no_of_queried_instances:         # True, as soon as nearest image is misclassified
                self._C_ben = True
            else:
                f[queried_instances] += label_puns                      # We only add the label puns, when we do not jump to stage 2 afterwards

        if self._C_ben:                                         # Stage 2 of query strategy
            mu_eff = 0.3                                                # Share of effektive population
            norm_threshold = torch.quantile(norms, mu_eff)
            queried_instances = norms < norm_threshold                  # We first only query mu_eff
            remaining_instances = norms >= norm_threshold
            no_of_queried_instances = queried_instances.sum()
            no_of_remaining_instances = remaining_instances.sum()

            preds = self._net(perturbed_image[queried_instances].reshape(no_of_queried_instances, 3, dim, dim)).argmax(1)
            label_puns = torch.where(preds == self._label, pun, 0)
            f[queried_instances] += label_puns
            self._adv_share = (torch.mean(label_puns/pun))

            if self._adv_share >= 0.75:                                 # Stage 3 of query strategy
                preds_rest = self._net(perturbed_image[remaining_instances].reshape(no_of_remaining_instances, 3, dim, dim)).argmax(1)    
                label_puns_rest = torch.where(preds_rest == self._label, pun, 0)        
                queries += (no_of_queried_instances.unsqueeze(0)+no_of_remaining_instances.unsqueeze(0))
                f[remaining_instances] += label_puns_rest
                self._adv_share = (self._adv_share*mu_eff) + (torch.mean(label_puns_rest/pun)*(1-mu_eff))
            else:
                queries += no_of_queried_instances.unsqueeze(0)

        self._query_counter = torch.cat((self._query_counter, queries))
        self._time = torch.cat((self._time, torch.tensor(time.time()-self._start_time, device=self._device).unsqueeze(0)))
        best_norm = f.min().unsqueeze(0)

        if len(self._best_norm) == 0:                                   # In the first generation, we always add the nearest instance, 
            self._best_norm = torch.cat((self._best_norm, best_norm))   # which must not be adversarial
            self._best_Instance = perturbed_image[torch.argmin(f)]
        elif (best_norm < torch.min(self._best_norm)) and (best_norm < pun):  # Only update if nearer and adversarial
            self._best_norm = torch.cat((self._best_norm, best_norm))
            self._best_Instance = perturbed_image[torch.argmin(f)]
        else:
            self._best_norm = torch.cat((self._best_norm, self._best_norm[-1].unsqueeze(0)))

        if not self._abl_c:                                             
            if self._C_ben and self._adv_share >= 0.95:         # When the share of adversarial examples is larger than t_adv = 0.95
                self._exp_update = self._e_sigma                        # Manually increase the exponential update for sigma
            else:
                self._exp_update = None

        solution.set_evals(f)   