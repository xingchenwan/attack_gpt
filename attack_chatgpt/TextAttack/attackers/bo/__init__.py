from ..classification import Classifier, ClassifierGoal
from ..genetic import GeneticAttacker
from typing import Optional, List
from ...text_process.tokenizer import Tokenizer
from ...attack_assist.substitute.word import WordSubstitute
import botorch
from typing import Dict, Any, Tuple
import gpytorch
from gpytorch.constraints import Interval
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.priors import GammaPrior
import torch
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from .encoder import BaseEncoder, SentenceEncoder
import numpy as np


class BayesOptAttacker(GeneticAttacker):

    def __init__(
            self,
            max_iters: int,
            auxiliary_model_class: BaseEncoder = SentenceEncoder,
            auxiliary_model_options: Optional[Dict[str, Any]] = None,
            batch_size: int = 1,
            n_init: int = 10,
            model_type: str = "gp",
            tokenizer: Optional[Tokenizer] = None,
            substitute: Optional[WordSubstitute] = None,
            lang=None,
            model_dir='',
            filter_words: List[str] = None,
            device: str = "cuda",
            dtype: torch.dtype = torch.float,
            acqf_options: Optional[Dict[str, Any]] = None,
            display_interval: int = 10,
    ):
        """
        BayesOpt attacker class.

        Args:
            max_iters: the maximum number of BO iterations
            auxiliary_model_class: the class to construct an encoding auxiliary model to obtain embedding.
                Signature: Callable[list[str]] -> torch.Tensor
            auxiliary_model_options: any keyword options to be passed to the auxiailisy_model_class constructor.
            batch_size: the number of queries to suggest, per BO iteration.
            n_init: the number of initial random points to sample prior to BO loop.
            model_type: the type of surrogate model. "gp" for vanilla GP model, "saasgp" for the SaaS-GP model for
                high dimensional modelling. It is significantly better in handling high-dimensional data, but can
                be much slower due the MCMC marginalization steps.
            tokenizer, substite, lang, model_dir, filter_words: see "genetic" attacker base object
            device, dtype: the device and dtype for the BO object.
            acqf_options: the settings regarding the acquisition function setting to be passed to the acqf constructor.
            display_interval: frequency in displaying the optimization progress.
        """
        assert max_iters >= n_init, f"max_iters is {max_iters} but n_init is {n_init}. Check again."
        assert model_type in [
            "gp", "saasgp"], f"model_type {model_type} is not implemented!"
        # TODO: implement other GP models, such as the ones with random projection for high-dimensionality

        super().__init__(
            pop_size=n_init,  # not used for Bayesopt attacker
            max_iters=max_iters,
            tokenizer=tokenizer,
            substitute=substitute,
            lang=lang,
            model_dir=model_dir,
            filter_words=filter_words
        )
        self.model_type = model_type
        self.batch_size = batch_size

        self.n_init = n_init
        self.n_init = n_init
        if device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tkwargs = {"device": device, "dtype": dtype}

        # Constuct the auxiliary model
        auxiliary_model_default_options = {
            "model_spec": "all-mpnet-base-v2"
        }
        auxiliary_model_default_options.update(auxiliary_model_options or {})
        auxiliary_model_default_options["device"] = self.tkwargs["device"]
        self.auxiliary_model = auxiliary_model_class(**auxiliary_model_default_options)

        # set the misc acqf_optim_options
        self.acqf_options = {
            # type of the acquisition function (default: ei)
            "acqf_type": "ei",
            # we perturb around `k`-best points seen so far as the starting point for acqf optim. This set the `k`
            "perturb_around_k_best": 5,
            # Number of restarts for acqf optimization
            "n_restarts": 5,
            # maximum number of acqf evaluations * per restart *
            "max_acqf_eval": 200,
            # Number of neighbours to subsample 
            "n_neighbours": 10
        }
        self.acqf_options.update(acqf_options or {})

        self.display_interval = display_interval

    def attack(self, victim: Classifier, x_orig, goal: ClassifierGoal):
        """Run main attack loop using Bayesian optimization attacker.
        """
        # string representation of original, after lower-casing
        x_orig_str = x_orig.lower()

        # Tokenize original
        x_orig = self.tokenizer.tokenize(x_orig_str)

        x_orig_pos = list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        # Get neighbours

        # Number of neighbors at each of the position in the sentence
        neighbours_nums = [
            self.get_neighbour_num(
                word, pos) if word not in self.filter_words else 0
            for word, pos in zip(x_orig, x_orig_pos)
        ]
        # the actual neighbors at each position
        # List[List[str]]: i-th element is the neighbors that differ at the i-th position
        neighbours = [
            self.get_neighbours(word, pos)
            if word not in self.filter_words
            else []
            for word, pos in zip(x_orig, x_orig_pos)
        ]

        if np.sum(neighbours_nums) == 0:
            return None
        w_select_probs = neighbours_nums / np.sum(neighbours_nums)

        # Generate a population (initial candidates) by perturbing randomly from
        # the original tokens
        x = [
            self.perturb(
                x_orig, x_orig, neighbours, w_select_probs, goal
            )
            for _ in range(self.n_init)
        ]
        # get the perturbed initial population
        x_strs = [self.tokenizer.detokenize(x_) for x_ in x]
        # get the vector embedding of initial
        y, is_successful = self._evaluate_objf(x_strs, victim=victim, goal=goal)
        if np.any(is_successful):
            success_index = is_successful == 1
            print(f"Attack success! {x_strs[success_index[0]]}")
            return x_strs[success_index[0]]
        
        with torch.no_grad():
            z = self.auxiliary_model(x_strs)

        # Initialize the surrogate model with y0 and z0
        z_torch = torch.tensor(z).to(**self.tkwargs)  # shape (N x D) D: Dim of embedding (typically 768)
        y_torch = torch.from_numpy(y).reshape(-1, 1).to(**self.tkwargs) # shape (N x 1)

        y_bests = y_torch.max().view(-1).cpu()
        x_bests = [x_strs[y_torch.argmax().item()]]

        for i in range(self.max_iters):
            if self.model_type == "gp":
                base_model_class = botorch.models.SingleTaskGP
            elif self.model_type == "saasgp":
                base_model_class = botorch.models.SaasFullyBayesianSingleTaskGP

            model = initialize_model(
                train_X=z_torch,
                train_Y=y_torch,
                base_model_class=base_model_class
            )
            acqf = get_acqf(
                model,
                train_Y=y_torch,
                label="ei",
            )
            # Best points seen so far -- we perturb the top-k points seen so far for
            # acquisition function optimization. `k` here is a hyperparameter.
            _, topk_ind = torch.topk(
                y_torch.reshape(-1), min(self.acqf_options["perturb_around_k_best"], y_torch.shape[0]))
            best_x = [x[i] for i in topk_ind]

            # generate the candidates by optimizing the acqf
            candidate_xs, candidate_xs_strs, candidate_zs = self._optimize_acqf(
                acqf=acqf,
                q=self.batch_size,
                x_baselines=best_x,
                x_orig=x_orig,
                neighbours=neighbours,
                w_select_probs=w_select_probs,
                goal=goal,
                Z_avoid=z_torch,
            )

            # query the victim model
            new_y, is_successful = self._evaluate_objf(candidate_xs_strs, victim=victim, goal=goal)
            if np.any(is_successful):
                success_index = is_successful == 1
                print(f"Attack success! {candidate_xs_strs[success_index[0]]}")
                return candidate_xs_strs[is_successful[0]]

            # Augment the data with new observations
            y_torch = torch.cat([y_torch, torch.from_numpy(new_y.reshape(1, -1)).to(y_torch)], dim=0)
            z_torch = torch.cat([z_torch, candidate_zs], dim=0)
            x += candidate_xs
            x_strs += candidate_xs_strs
            
            # record the best values seen so far
            y_best = y_torch.max().view(-1).cpu()
            y_bests = torch.cat([y_bests, y_best.view(-1)], dim=0)
            x_bests.append(x_strs[int(y_torch.argmax())])

            if i % self.display_interval == 0 and i:
                print(f"Iteration {i} / {self.max_iters}: Best reward: {y_bests[-1].item()}. Best candidate: {x_bests[-1]}")

        return None # failed


    def _evaluate_objf(self, queries: List[str], victim: Classifier, goal: ClassifierGoal):
        """Evaluate the objective function value.

        Args:
            queries: the (untokenized) queries that should be fed into the victim model of 
                shape (N, )
            victim: the victim classifier model
            goal: a ClassifierGoal object
        Retrusn:
            score: the objective function value. Assumed this value should be maximized
            is_successful: binary np.ndarray. Whether we have achieved a succesful attack.
            Both return values should have shape (N,) (identical to that of `queries`.)
        """
        victim_preds = victim.get_prob(queries)
        if goal.targeted:
            obj_val = victim_preds[:, goal.target]
            is_successful = victim_preds.argmax(axis=-1) == goal.target
        else:
            obj_val = -victim_preds[:, goal.target]
            is_successful = victim_preds.argmax(axis=-1) != goal.target
        return obj_val, np.array(is_successful).astype(int)

    def _optimize_acqf(
            self,
            acqf: botorch.acquisition.AcquisitionFunction,
            q: int,
            x_baselines,
            x_orig,
            neighbours: List[List[str]],
            w_select_probs,
            goal: ClassifierGoal,
            Z_avoid: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[str]], List[str], torch.Tensor]:
        """Routine to optimize the acquisition function.

        Args:
            acqf: a botorch.acquisition.AcquisitionFunction object. The acquisition function of which
                we aim to maximize value.
            q: batch size.
            x_baselines: the sequence of candidates that have achieved the best values so far, the
                perturbed versions of which will be used as the starting point for local search
            x_orig: the original queries after tokenization
            neighbours: the list of list of strings denoting the neighbours of the query at each of
                its position.
            w_select_probs: the probability of selection at each position. passed to self.perturb.
            goal: the classifier goal.
        
        Returns:
            candidate_xs: the tokenized candidates to evaluate next
            candidate_xs_strs: the untokenized (i.e. string representation) of `candidate_xs`.
            candidate_zs: the embedding of `candidate_xs` and `candidate_xs_strs` under the auxiliary encoder.
        """
        base_X_pending = acqf.X_pending if q > 1 else None
        n_restarts = self.acqf_options["n_restarts"]
        Z_avoid = Z_avoid if Z_avoid is not None else None

        # Three different representation of returned values
        candidate_xs = []  # the tokenized candidates
        candidate_xs_strs = []  # the de-tokenized candidates (i.e. as strings)
        # the candidates' sentence embedding from the encoder (auxiliary model)
        candidate_zs = []
        
        # Loop over the batch size
        for _ in range(q):

            X0, X0_str, Z0 = [], [], []
            while len(X0) < n_restarts:
                # randomly perturb one of the baseline points (typically the best points seen so far) as starting point
                x0 = self.perturb(x_baselines[np.random.choice(len(x_baselines))],
                                  x_orig, neighbours, w_select_probs, goal)
                x0_str = self.tokenizer.detokenize(x0)
                with torch.no_grad():
                    z0 = torch.from_numpy(self.auxiliary_model([x0_str])).reshape(1, -1).to(**self.tkwargs)
                if Z_avoid is None:
                    Z_avoid = torch.zeros(0, z0.shape[-1]).to(**self.tkwargs)
                z0 = filter_invalid(z0, Z_avoid)
                if not z0.shape[0]:
                    continue
                X0.append(x0)
                X0_str.append(x0_str)
                Z0.append(z0)
            Z0 = torch.cat(Z0).to(**self.tkwargs)
            best_zs = Z0
            best_xs, best_xs_str = [None] * Z0.shape[0], [None] * Z0.shape[0]

            with torch.no_grad():
                best_acqvals = acqf(Z0.unsqueeze(1))

            # Iterate over n_restarts
            for j, z0 in enumerate(Z0):
                curr_z, curr_f = z0.clone(), best_acqvals[j]
                curr_x = X0[j]
                curr_x_str = X0_str[j]

                n_evals_left = self.acqf_options["max_acqf_eval"]

                while n_evals_left > 0:
                    # perturb the current value
                    neighbor_x = [self.perturb(curr_x, x_orig, neighbours, w_select_probs, goal)
                                  for _
                                  in range(self.acqf_options["n_neighbours"])]
                    neighbor_x_str = [
                        self.tokenizer.detokenize(x) for x in neighbor_x]
                    neighbor_z = self.auxiliary_model(neighbor_x_str)
                    neighbor_z = torch.from_numpy(neighbor_z).to(**self.tkwargs)
                    neighbor_acqf = acqf(neighbor_z.unsqueeze(1))

                    n_evals_left -= neighbor_z.shape[0]
                    if neighbor_acqf.max() > curr_f:
                        indbest = neighbor_acqf.argmax()
                        curr_z = neighbor_z[indbest]
                        curr_x = neighbor_x[indbest]
                        curr_x_str = neighbor_x_str[indbest]
                        curr_f = neighbor_acqf.max()

                best_zs[j, :], best_acqvals[j] = curr_z, curr_f
                best_xs[j] = curr_x
                best_xs_str[j] = curr_x_str

            best_idx = best_acqvals.argmax()
            candidate_xs.append(best_xs[best_idx])
            candidate_xs_strs.append(best_xs_str[best_idx])
            candidate_zs.append(best_zs[best_idx].reshape(1, -1))

            # Set pending points
            candidate_z_torch = torch.cat(candidate_zs, dim=-2)
            if q > 1:
                acqf.set_X_pending(
                    torch.cat([base_X_pending, candidate_z_torch], dim=-2)
                    if base_X_pending is not None
                    else candidate_z_torch
                )
                Z_avoid = (
                    torch.cat([Z_avoid, candidate_z_torch], dim=-2)
                )
        # Reset acq_func to original X_pending state
        if q > 1:
            if hasattr(acqf, "set_X_pending"):
                acqf.set_X_pending(base_X_pending)
        return candidate_xs, candidate_xs_strs, candidate_z_torch

    def perturb(
        self, x_cur, x_orig, neighbours, w_select_probs, goal: ClassifierGoal,
        clsf: Optional[Classifier] = None,
        ensure_increasing_score: bool = False,
    ):
        x_len = len(x_cur)
        num_mods = 0
        for i in range(x_len):
            if x_cur[i] != x_orig[i]:
                num_mods += 1
        mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        if num_mods < np.sum(
            np.sign(w_select_probs)
        ):  # exists at least one indx not modified
            while x_cur[mod_idx] != x_orig[mod_idx]:  # already modified
                mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[
                    0
                ]  # random another indx
        return self.select_best_replacements(
            mod_idx,
            neighbours[mod_idx],
            x_cur,
            x_orig,
            goal,
            clsf=clsf,
            ensure_increasing_score=ensure_increasing_score
        )

    def select_best_replacements(
        self,
        indx,
        neighbours,
        x_cur,
        x_orig,
        goal: ClassifierGoal,
        clsf: Optional[Classifier] = None,
        ensure_increasing_score: bool = False
    ):
        """
        Select the a replacement that modify the original sentence at sentence position
        `index`. When `ensure_increasing_score` is True, the victim model is queried and
        only the perturbation that leads to the largest increase in score is selected.
        Otherwise we randomly choose a perturbation (makes more sense in BO where sample
        efficiency is valued).

        Modified version of the one used in genetic. This does not necessarily query the
        `clsf` when selecting the exact candidate to query next.
        """
        if ensure_increasing_score and clsf is None:
            raise ValueError(
                "if ensure_increasing_score is enabled, clsf must be supplied!")

        def do_replace(word):
            ret = x_cur.copy()
            ret[indx] = word
            return ret
        new_list = []
        rep_words = []
        for word in neighbours:
            if word != x_orig[indx]:
                new_list.append(do_replace(word))
                rep_words.append(word)
        if len(new_list) == 0:
            return x_cur
        new_list.append(x_cur)
        if not ensure_increasing_score:
            return new_list[np.random.choice(len(new_list))]

        pred_scores = clsf.get_prob(self.make_batch(new_list))[:, goal.target]
        if goal.targeted:
            new_scores = pred_scores[:-1] - pred_scores[-1]
        else:
            new_scores = pred_scores[-1] - pred_scores[:-1]

        if np.max(new_scores) > 0:
            return new_list[np.argmax(new_scores)]
        else:
            return x_cur


def get_kernel(use_ard: bool = False):
    """Helper method to initialize the kernels"""
    kernels = []
    if not use_ard:
        kernels.append(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=None
            )
        )
    else:
        kernels.append(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
            )
        )

    kernel = kernels[0]
    if len(kernels) > 1:
        for k in kernels[1:]:
            kernel *= k
    return gpytorch.kernels.ScaleKernel(kernel)


def initialize_model(
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        base_model_class=None,
        fit_model: bool = True,
        verbose: bool = False,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        covar_module_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
):
    """Initialize a GP model."""
    if train_Y.ndim == 1:
        train_Y = train_Y.reshape(-1, 1)
    model_kwargs = []
    # define the default values
    optimizer_kwargs_ = {
        # default settings for fitting the SAAS-GP model
        "warmup_steps": 256,
        "num_samples": 128,
        "thinning": 16
    }
    optimizer_kwargs_.update(optimizer_kwargs or {})
    covar_module_kwargs = covar_module_kwargs or {}

    # initialize the model classes -- note that each dimension of output can have a different model class
    if base_model_class is None:
        base_model_class = SingleTaskGP

    # initialize the covariance module
    if base_model_class != SaasFullyBayesianSingleTaskGP:
        covar_module = covar_module or get_kernel(**covar_module_kwargs)
    else:
        covar_module = None     # for Saasmodel we don't need to specify the covariance module

    for i in range(train_Y.shape[-1]):
        model_kwargs.append(
            {
                "train_X": train_X,
                "train_Y": train_Y[..., i: i + 1],
                "outcome_transform": Standardize(m=1),
            }
        )
        if base_model_class != SaasFullyBayesianSingleTaskGP:
            model_kwargs[i]["covar_module"] = covar_module
            model_kwargs[i]["likelihood"] = GaussianLikelihood(
                noise_prior=GammaPrior(0.9, 10.0),
                noise_constraint=Interval(1e-7, 1e-3)
            )
    models = [base_model_class(**model_kwargs[i])
              for i in range(train_Y.shape[-1])]
    if len(models) > 1:
        model = ModelListGP(*models).to(device=train_X.device)
    else:
        model = models[0].to(device=train_X.device)

    if verbose:
        print(model)

    # fit the model
    if fit_model:
        if len(models) == 1:
            if base_model_class == SaasFullyBayesianSingleTaskGP:
                n_attempt = 0
                while n_attempt < 3:
                    try:
                        fit_fully_bayesian_model_nuts(model,
                                                      warmup_steps=optimizer_kwargs_[
                                                          "warmup_steps"],
                                                      num_samples=optimizer_kwargs_[
                                                          "num_samples"],
                                                      thinning=optimizer_kwargs_[
                                                          "thinning"],
                                                      disable_progbar=True)
                        break
                    except Exception as e:
                        n_attempt += 1
                if n_attempt >= 3:
                    raise ValueError(
                        f"Fitting SAASGP Failed with error message {e}")
            else:
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                    model.likelihood, model).to(device=train_X.device)
                fit_gpytorch_torch(mll)
        else:
            # if there is at least one model that contains a SAAS-GP model -- note that SAAS-GP model needs to
            # be fitted differently.
            if base_model_class == SaasFullyBayesianSingleTaskGP:
                for i in range(train_Y.shape[-1]):
                    if base_model_class == SaasFullyBayesianSingleTaskGP:
                        fit_fully_bayesian_model_nuts(model.models[i],
                                                      warmup_steps=optimizer_kwargs_[
                                                          "warmup_steps"],
                                                      num_samples=optimizer_kwargs_[
                                                          "num_samples"],
                                                      thinning=optimizer_kwargs_[
                                                          "thinning"],
                                                      disable_progbar=True)
                    else:
                        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.models[i].likelihood, model.models[i]).to(
                            device=train_X.device)
                        fit_gpytorch_torch(mll)
            else:
                mll = gpytorch.mlls.SumMarginalLogLikelihood(
                    model.likelihood, model).to(device=train_X.device)
                n_attempt = 0
                while n_attempt < 3:
                    try:
                        fit_gpytorch_torch(mll, options={"disp": False})
                        break
                    except Exception as e:
                        n_attempt += 1
                    if n_attempt >= 3:
                        print(
                            f"Fitting model failed after {n_attempt} number of attempts with error {e}!")

    return model


def get_acqf(model,
             train_Y: torch.Tensor,
             label: str = None,
             ):
    """Get the acquisition function from botorch"""
    label = label or "ei"
    if label == "ei":
        acq_func = botorch.acquisition.qExpectedImprovement(
            model, train_Y.max())
    else:
        raise NotImplementedError(
            f"acquisition function {label} is not implemented!")
    return acq_func


def filter_invalid(X: torch.Tensor, X_avoid: torch.Tensor, return_index: bool = False):
    """Remove all occurences of `X_avoid` from `X`."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X_avoid.ndim == 1:
        X_avoid = X_avoid.reshape(-1, 1)
    idx = ~(X == X_avoid.unsqueeze(-2)).all(dim=-1).any(dim=-2)
    ret = X[idx]
    if X.ndim == 1:
        ret = ret.squeeze(1)
    if return_index:
        return ret, idx.nonzero().squeeze(-1).tolist()
    return ret