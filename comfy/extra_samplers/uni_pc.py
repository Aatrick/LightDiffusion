#code taken from: https://github.com/wl-zhao/UniPC and modified

import torch

from tqdm.auto import trange


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.

    We support four types of the diffusion model by setting `model_type`:

        1. "noise": noise prediction model. (Trained by predicting noise).

        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).

        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].

            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
                arXiv preprint arXiv:2202.00512 (2022).
            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
                arXiv preprint arXiv:2210.02303 (2022).
    
        4. "score": marginal score function. (Trained by denoising score matching).
            Note that the score function and the noise prediction model follows a simple relationship:
            ```
                noise(x_t, t) = -sigma_t * score(x_t, t)
            ```

    We support three types of guided sampling by DPMs by setting `guidance_type`:
        1. "uncond": unconditional sampling by DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

        2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            `` 

            The input `classifier_fn` has the following format:
            ``
                classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
            ``

            [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

        3. "classifier-free": classifier-free guidance sampling by conditional DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            `` 
            And if cond == `unconditional_condition`, the model output is the unconditional DPM output.

            [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
                arXiv preprint arXiv:2207.12598 (2022).
        

    The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)
    or continuous-time labels (i.e. epsilon to T).

    We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:
    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return noise_pred(model, x, t_input, **model_kwargs)         
    ``
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    ===============================================================

    Args:
        model: A diffusion model with the corresponding format described above.
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        model_type: A `str`. The parameterization type of the diffusion model.
                    "noise" or "x_start" or "v" or "score".
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
        guidance_type: A `str`. The type of the guidance for sampling.
                    "uncond" or "classifier" or "classifier-free".
        condition: A pytorch tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
        unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.
                    Only used for "classifier-free" guidance type.
        guidance_scale: A `float`. The scale for the guided sampling.
        classifier_fn: A classifier function. Only used for the classifier guidance.
        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the continuous time as the inputs.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        t_input = get_model_input_time(t_continuous)
        output = model(x, t_input, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return (x - expand_dims(alpha_t, dims) * output) / expand_dims(sigma_t, dims)
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return expand_dims(alpha_t, dims) * output + expand_dims(sigma_t, dims) * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return -expand_dims(sigma_t, dims) * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, dims=cond_grad.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class UniPC:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        predict_x0=True,
        thresholding=False,
        max_val=1.,
        variant='bh1',
        noise_mask=None,
        masked_image=None,
        noise=None,
    ):
        """Construct a UniPC. 

        We support both data_prediction and noise_prediction.
        """
        self.model = model_fn
        self.noise_schedule = noise_schedule
        self.variant = variant
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val
        self.noise_mask = noise_mask
        self.masked_image = masked_image
        self.noise = noise

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        if self.noise_mask is not None:
            return self.model(x, t) * self.noise_mask
        else:
            return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with thresholding).
        """
        noise = self.noise_prediction_fn(x, t)
        dims = x.dim()
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - expand_dims(sigma_t, dims) * noise) / expand_dims(alpha_t, dims)
        if self.thresholding:
            p = 0.995   # A hyperparameter in the paper of "Imagen" [1].
            s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
            s = expand_dims(torch.maximum(s, self.max_val * torch.ones_like(s).to(s.device)), dims)
            x0 = torch.clamp(x0, -s, s) / s
        if self.noise_mask is not None:
            x0 = x0 * self.noise_mask + (1. - self.noise_mask) * self.masked_image
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def multistep_uni_pc_update(self, x, model_prev_list, t_prev_list, t, order, **kwargs):
        if len(t.shape) == 0:
            t = t.view(-1)
        if 'bh' in self.variant:
            return self.multistep_uni_pc_bh_update(x, model_prev_list, t_prev_list, t, order, **kwargs)
        else:
            assert self.variant == 'vary_coeff'
            return self.multistep_uni_pc_vary_update(x, model_prev_list, t_prev_list, t, order, **kwargs)

    def multistep_uni_pc_vary_update(self, x, model_prev_list, t_prev_list, t, order, use_corrector=True):
        print(f'using unified predictor-corrector with order {order} (solver type: vary coeff)')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        K = len(rks)
        # build C matrix
        C = []

        col = torch.ones_like(rks)
        for k in range(1, K + 1):
            C.append(col)
            col = col * rks / (k + 1) 
        C = torch.stack(C, dim=1)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1) # (B, K)
            C_inv_p = torch.linalg.inv(C[:-1, :-1])
            A_p = C_inv_p

        if use_corrector:
            print('using corrector')
            C_inv = torch.linalg.inv(C)
            A_c = C_inv

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_ks = []
        factorial_k = 1
        h_phi_k = h_phi_1
        for k in range(1, K + 2):
            h_phi_ks.append(h_phi_k)
            h_phi_k = h_phi_k / hh - 1 / factorial_k
            factorial_k *= (k + 1)

        model_t = None
        if self.predict_x0:
            x_t_ = (
                sigma_t / sigma_prev_0 * x
                - alpha_t * h_phi_1 * model_prev_0
            )
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_p[k])
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = (model_t - model_prev_0)
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_c[k][:-1])
                x_t = x_t - alpha_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        else:
            log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
            x_t_ = (
                (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * h_phi_1) * model_prev_0
            )
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_p[k])
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = (model_t - model_prev_0)
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_c[k][:-1])
                x_t = x_t - sigma_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        return x_t, model_t

    def multistep_uni_pc_bh_update(self, x, model_prev_list, t_prev_list, t, order, x_t=None, use_corrector=True):
        # print(f'using unified predictor-corrector with order {order} (solver type: B(h))')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)
        dims = x.dim()

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h[0] if self.predict_x0 else h[0]
        h_phi_1 = torch.expm1(hh) # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == 'bh1':
            B_h = hh
        elif self.variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()
            
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i 

        R = torch.stack(R)
        b = torch.tensor(b, device=x.device)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1) # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            # print('using corrector')
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        if self.predict_x0:
            x_t_ = (
                expand_dims(sigma_t / sigma_prev_0, dims) * x
                - expand_dims(alpha_t * h_phi_1, dims)* model_prev_0
            )

            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - expand_dims(alpha_t * B_h, dims) * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = (model_t - model_prev_0)
                x_t = x_t_ - expand_dims(alpha_t * B_h, dims) * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = (
                expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                - expand_dims(sigma_t * h_phi_1, dims) * model_prev_0
            )
            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - expand_dims(sigma_t * B_h, dims) * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = (model_t - model_prev_0)
                x_t = x_t_ - expand_dims(sigma_t * B_h, dims) * (corr_res + rhos_c[-1] * D1_t)
        return x_t, model_t


    def sample(self, x, timesteps, order=3,
        method='singlestep', lower_order_final=True, callback=None, disable_pbar=False
    ):
        # t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        # t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        steps = len(timesteps) - 1
        if method == 'multistep':
            assert steps >= order
            # timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
            # with torch.no_grad():
            for step_index in trange(steps, disable=disable_pbar):
                if self.noise_mask is not None:
                    x = x * self.noise_mask + (1. - self.noise_mask) * (self.masked_image * self.noise_schedule.marginal_alpha(timesteps[step_index]) + self.noise * self.noise_schedule.marginal_std(timesteps[step_index]))
                if step_index == 0:
                    vec_t = timesteps[0].expand((x.shape[0]))
                    model_prev_list = [self.model_fn(x, vec_t)]
                    t_prev_list = [vec_t]
                elif step_index < order:
                    init_order = step_index
                # Init the first `order` values by lower order multistep DPM-Solver.
                # for init_order in range(1, order):
                    vec_t = timesteps[init_order].expand(x.shape[0])
                    x, model_x = self.multistep_uni_pc_update(x, model_prev_list, t_prev_list, vec_t, init_order, use_corrector=True)
                    if model_x is None:
                        model_x = self.model_fn(x, vec_t)
                    model_prev_list.append(model_x)
                    t_prev_list.append(vec_t)
                else:
                    extra_final_step = 0
                    if step_index == (steps - 1):
                        extra_final_step = 1
                    for step in range(step_index, step_index + 1 + extra_final_step):
                        vec_t = timesteps[step].expand(x.shape[0])
                        if lower_order_final:
                            step_order = min(order, steps + 1 - step)
                        else:
                            step_order = order
                        # print('this step order:', step_order)
                        if step == steps:
                            # print('do not run corrector at the last step')
                            use_corrector = False
                        else:
                            use_corrector = True
                        x, model_x =  self.multistep_uni_pc_update(x, model_prev_list, t_prev_list, vec_t, step_order, use_corrector=use_corrector)
                        for i in range(order - 1):
                            t_prev_list[i] = t_prev_list[i + 1]
                            model_prev_list[i] = model_prev_list[i + 1]
                        t_prev_list[-1] = vec_t
                        # We do not need to evaluate the final model value.
                        if step < steps:
                            if model_x is None:
                                model_x = self.model_fn(x, vec_t)
                            model_prev_list[-1] = model_x
                if callback is not None:
                    callback(step_index, model_prev_list[-1], x, steps)
        else:
            raise NotImplementedError()
        # if denoise_to_zero:
        #     x = self.denoise_to_zero_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
        return x


#############################################################
# other utility functions
#############################################################
def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]


class SigmaConvert:
    schedule = ""
    def marginal_log_mean_coeff(self, sigma):
        return 0.5 * torch.log(1 / ((sigma * sigma) + 1))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

def predict_eps_sigma(model, input, sigma_in, **kwargs):
    sigma = sigma_in.view(sigma_in.shape[:1] + (1,) * (input.ndim - 1))
    input = input * ((sigma ** 2 + 1.0) ** 0.5)
    return  (input - model(input, sigma_in, **kwargs)) / sigma


def sample_unipc(model, noise, image, sigmas, sampling_function, max_denoise, extra_args=None, callback=None, disable=False, noise_mask=None, variant='bh1'):
        timesteps = sigmas.clone()
        if sigmas[-1] == 0:
            timesteps = sigmas[:]
            timesteps[-1] = 0.001
        else:
            timesteps = sigmas.clone()
        ns = SigmaConvert()

        if image is not None:
            img = image * ns.marginal_alpha(timesteps[0])
            if max_denoise:
                noise_mult = 1.0
            else:
                noise_mult = ns.marginal_std(timesteps[0])
            img += noise * noise_mult
        else:
            img = noise

        model_type = "noise"

        model_fn = model_wrapper(
            lambda input, sigma, **kwargs: predict_eps_sigma(model, input, sigma, **kwargs),
            ns,
            model_type=model_type,
            guidance_type="uncond",
            model_kwargs=extra_args,
        )

        order = min(3, len(timesteps) - 2)
        uni_pc = UniPC(model_fn, ns, predict_x0=True, thresholding=False, noise_mask=noise_mask, masked_image=image, noise=noise, variant=variant)
        x = uni_pc.sample(img, timesteps=timesteps, skip_type="time_uniform", method="multistep", order=order, lower_order_final=True, callback=callback, disable_pbar=disable)
        x /= ns.marginal_alpha(timesteps[-1])
        return x
