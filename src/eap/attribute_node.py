from typing import Callable, Union, Optional, Literal
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
from einops import einsum

from .graph import Graph
from .utils import tokenize_plus, compute_mean_activations
from .evaluate import evaluate_baseline, evaluate_graph


def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores: Optional[Tensor], neuron:bool=False):
    """Makes a matrix, and hooks to fill it and the score matrix up

    Args:
        model (HookedTransformer): model to attribute
        graph (Graph): graph to attribute
        batch_size (int): size of the particular batch you're attributing
        n_pos (int): size of the position dimension
        scores (Tensor): The scores tensor you intend to fill. If you pass in None, we assume that you're using these hooks / matrices for evaluation only (so don't use the backwards hooks!)

    Returns:
        Tuple[Tuple[List, List, List], Tensor]: The final tensor ([batch, pos, n_src_nodes, d_model]) stores activation differences, 
        i.e. corrupted - clean activations. The first set of hooks will add in the activations they are run on (run these on corrupted input), 
        while the second set will subtract out the activations they are run on (run these on clean input). 
        The third set of hooks will compute the gradients and update the scores matrix that you passed in. 
    """
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)

    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
        
    # Fills up the activation difference matrix. In the default case (not separate_activations), 
    # we add in the corrupted activations (add = True) and subtract out the clean ones (add=False)
    # In the separate_activations case, we just store them in two halves of the matrix. Less efficient, 
    # but necessary for models with Gemma's architecture.
    def activation_hook(index, activations:torch.Tensor, hook: HookPoint, add:bool=True):
        acts = activations.detach()
        try:
            if add:
                activation_difference[:, :, index] += acts
            else:
                activation_difference[:, :, index] -= acts

        except RuntimeError as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e
    
    def gradient_hook(fwd_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, hook: HookPoint):
        """Takes in a gradient and uses it and activation_difference 
        to compute an update to the score matrix

        Args:
            fwd_index (Union[slice, int]): The forward index of the (src) node
            bwd_index (Union[slice, int]): The backward index of the (dst) node
            gradients (torch.Tensor): The gradients of this backward pass 
            hook (_type_): (unused)

        """
        grads = gradients.detach()
        try:
            if neuron:
                s = einsum(activation_difference[:, :, fwd_index], grads,'batch pos ... hidden, batch pos ... hidden -> ... hidden')
            else:
                s = einsum(activation_difference[:, :, fwd_index], grads,'batch pos ... hidden, batch pos ... hidden -> ...')
            scores[fwd_index] += s
        except RuntimeError as e:
            print(hook.name, activation_difference.size(), activation_difference.device, grads.size(), grads.device)
            print(fwd_index, bwd_index, scores.size())
            raise e

    node = graph.nodes['input']
    fwd_index = graph.forward_index(node)
    fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
    fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
    bwd_hooks.append((node.out_hook, partial(gradient_hook, fwd_index, fwd_index)))
    
    for layer in range(graph.cfg['n_layers']):
        node = graph.nodes[f'a{layer}.h0']
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.out_hook, partial(gradient_hook, fwd_index, fwd_index)))

        node = graph.nodes[f'm{layer}']
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.out_hook, partial(gradient_hook, fwd_index, fwd_index)))

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def get_scores_exact(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], 
                     intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                     intervention_dataloader: Optional[DataLoader]=None, quiet=False):
    """Gets scores via exact patching, by repeatedly calling evaluate graph.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): the intervention to use. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
        quiet (bool, optional): _description_. Defaults to False.
    """

    graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
    graph.nodes_in_graph[:] = True
    baseline = evaluate_baseline(model, dataloader, metric).mean().item()
    nodes = graph.nodes.values() if quiet else tqdm(graph.nodes.values())
    for node in nodes:
        for edge in node.child_edges:
            edge.in_graph = False
        intervened_performance = evaluate_graph(model, graph, dataloader, metric, intervention=intervention, 
                                                intervention_dataloader=intervention_dataloader, quiet=True, skip_clean=True).mean().item()
        node.score = intervened_performance - baseline
        for edge in node.child_edges:
            edge.in_graph = True

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.nodes_scores


def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], 
                   intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                   intervention_dataloader: Optional[DataLoader]=None, quiet:bool=False, neuron:bool=False):
    """Gets edge attribution scores using EAP.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)    
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)    

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
            elif 'mean' in intervention:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores

def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                      steps=30, quiet:bool=False, neuron:bool=False):
    """Gets edge attribution scores using EAP with integrated gradients.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        steps (int, optional): number of IG steps. Defaults to 30.
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)    
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations 
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

            input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        # + activations * 0  will cause a backwards pass on new_input
        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) + activations * 0
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_eap_ig_local(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
                            steps=30, quiet:bool=False, neuron:bool=False):
    """Node-level EAP-IG (inputs) using the LOCAL activation increment inside the IG sum.

    Standard EAP-IG-inputs scores a node by (a^corrupted - a^clean) . (1/m) sum_k grad(alpha_k),
    pulling the endpoint activation difference out of the integral. That is exact only if the
    node's activation is linear in alpha; but only the INPUT embeddings are interpolated linearly,
    so intermediate activations follow a nonlinear path. This variant instead accumulates the
    realized per-step increment  sum_k grad(alpha_k) . (a(alpha_{k-1}) - a(alpha_k)), i.e. the
    proper Riemann sum along the actual activation trajectory. Gradients are sampled on the same
    grid as node EAP-IG-inputs (alpha = k/steps, k=1..steps), so with steps=1 it is identical.
    """
    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        # activation_difference is what the bwd hooks dot against grads; we overwrite it each
        # step (via the capture hooks below) with the LOCAL increment (prev - cur) per node.
        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        (fwd_hooks_add, _, _), acts_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, None, neuron=neuron)
        (_, fwd_hooks_sub, _), neg_acts_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, None, neuron=neuron)
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_add):
                _ = model(corrupted_tokens, attention_mask=attention_mask)        # acts_corrupted = +corrupted
            with model.hooks(fwd_hooks=fwd_hooks_sub):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)  # neg_acts_clean = -clean
        acts_clean = -neg_acts_clean

        input_idx = graph.forward_index(graph.nodes['input'])
        input_corrupted = acts_corrupted[:, :, input_idx].clone()
        input_clean = acts_clean[:, :, input_idx].clone()

        prev_acts = acts_corrupted.clone()   # a(alpha_0) = corrupted
        cur_acts = acts_corrupted.clone()

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                alpha = k / steps
                new_input = input_corrupted + alpha * (input_clean - input_corrupted) + activations * 0
                cur_acts[:, :, input_idx] = new_input.detach()
                activation_difference[:, :, input_idx] = prev_acts[:, :, input_idx] - new_input.detach()
                return new_input
            return hook_fn

        def capture_hook(index, activations, hook):
            acts = activations.detach()
            cur_acts[:, :, index] = acts
            activation_difference[:, :, index] = prev_acts[:, :, index] - acts

        capture_fwd_hooks = []
        for layer in range(graph.cfg['n_layers']):
            for node in (graph.nodes[f'a{layer}.h0'], graph.nodes[f'm{layer}']):
                capture_fwd_hooks.append((node.out_hook, partial(capture_hook, graph.forward_index(node))))

        for step in range(1, steps + 1):
            fwd_hooks = [(graph.nodes['input'].out_hook, input_interpolation_hook(step))] + capture_fwd_hooks
            with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()
            prev_acts = cur_acts.clone()

    # No /steps: the per-step increment already carries the 1/steps factor.
    scores /= total_items
    return scores


class ShapleyElementwiseMult(torch.autograd.Function):
    """Elementwise multiply with the RelP/Shapley half-rule: each branch gets 0.5 of the
    incoming gradient (so the product's relevance is split evenly instead of double-counted)."""
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return 0.5 * grad_output * y, 0.5 * grad_output * x


class HalfGrad(torch.autograd.Function):
    """Identity forward, 0.5x gradient backward. Applied to a bilinear matmul's OUTPUT this
    is equivalent to Transluce's ShapleyMatmul half-rule on its inputs: for z = x @ y, scaling
    grad_z by 0.5 yields grad_x, grad_y each halved. Used for the AttnRLP QK and OV matmuls."""
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return 0.5 * grad_output


class ShapleySoftmax(torch.autograd.Function):
    """LRP/Shapley-style softmax backward (Transluce AttnRLP). Forward = softmax; backward
    redistributes relevance proportional to the softmax output and divides by the pre-softmax
    scores: grad_x = (sum_j grad_j * p_j) * p_i / scores_i."""
    @staticmethod
    def forward(ctx, x):
        result = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
        ctx.save_for_backward(x, result)
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        logits, result = ctx.saved_tensors
        with torch.no_grad():
            R_l = grad_output.to(torch.float32) * result
            total_R = R_l.sum(-1, keepdim=True)
            grad_x = (total_R * result) / logits
        return grad_x.to(logits.dtype)


def _relp_act_coeff(act_fn, pre):
    """Secant linearization of a (gated) activation: coeff = act(pre)/pre, detached.
    For SiLU this equals sigmoid(pre) exactly (silu(z)=z*sigmoid(z)), matching the
    Transluce RelP gate rule; gate_act = pre*coeff preserves the forward value while
    letting the gradient flow through `pre` with the nonlinearity treated as constant."""
    a = act_fn(pre)
    safe = torch.where(pre.abs() < 1e-6, torch.ones_like(pre), pre)
    return (a / safe).detach()


def build_relp_fwd_hooks(model: HookedTransformer, use_norm=True, use_mlp=True, use_qk=True, shapley_attn=False):
    """Forward hooks implementing RelP's three backward modifications (forward values
    unchanged; only the gradient is altered):
      (1) every norm scale is detached -> normalization treated as a constant scaling;
      (2) the gated-MLP gate activation is secant-linearized and the gate x up product
          uses the half-rule (non-gated MLPs just get the linearized activation);
      (3) the attention pattern is detached -> gradient flows only through OV (V path).
    """
    import os
    # explicit args set the default; env vars override (for ablation sweeps)
    use_norm = os.environ.get('RELP_NORM', '1' if use_norm else '0') == '1'
    use_mlp = os.environ.get('RELP_MLP', '1' if use_mlp else '0') == '1'
    use_qk = os.environ.get('RELP_QK', '1' if use_qk else '0') == '1'
    hooks = []
    for name in list(model.hook_dict.keys()):
        if (use_norm and name.endswith('.hook_scale')) or (use_qk and name.endswith('.hook_pattern')):
            hooks.append((name, lambda t, hook: t.detach()))

    # AttnRLP attention rules: half-rule on the QK and OV matmuls (HalfGrad on their outputs)
    # plus the LRP softmax backward (ShapleySoftmax). hook_attn_scores fires pre-softmax, so we
    # capture the (scaled+masked) scores there and rebuild the pattern with ShapleySoftmax.
    if shapley_attn:
        def make_attn_hooks(holder):
            def cap_scores(t, hook):
                s = HalfGrad.apply(t)        # QK matmul half-rule (grad x0.5 to Q and K)
                holder['scores'] = s
                return s
            def shapley_pattern(t, hook):
                return ShapleySoftmax.apply(holder['scores'])   # LRP softmax backward
            def half_z(t, hook):
                return HalfGrad.apply(t)     # OV matmul half-rule (grad x0.5 to pattern and V)
            return cap_scores, shapley_pattern, half_z
        for l in range(model.cfg.n_layers):
            cap, pat, hz = make_attn_hooks({})
            hooks.append((f'blocks.{l}.attn.hook_attn_scores', cap))
            hooks.append((f'blocks.{l}.attn.hook_pattern', pat))
            hooks.append((f'blocks.{l}.attn.hook_z', hz))

    if not use_mlp:
        return hooks

    def make_store(holder, key):
        def fn(t, hook):
            holder[key] = t
            return t
        return fn

    def make_relp_post(holder, act_fn, b_in, gated):
        def fn(post, hook):
            pre = holder['pre']
            gate_act = pre * _relp_act_coeff(act_fn, pre)
            if gated:
                out = ShapleyElementwiseMult.apply(gate_act, holder['pre_linear'])
                if b_in is not None:
                    out = out + b_in
                return out
            return gate_act
        return fn

    for l in range(model.cfg.n_layers):
        mlp = model.blocks[l].mlp
        gated = hasattr(mlp, 'W_gate')
        b_in = getattr(mlp, 'b_in', None)
        holder = {}
        hooks.append((f'blocks.{l}.mlp.hook_pre', make_store(holder, 'pre')))
        if gated:
            hooks.append((f'blocks.{l}.mlp.hook_pre_linear', make_store(holder, 'pre_linear')))
        hooks.append((f'blocks.{l}.mlp.hook_post', make_relp_post(holder, mlp.act_fn, b_in, gated)))
    return hooks


def get_scores_relp(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
                    quiet: bool = False, neuron: bool = False, relp_hooks: bool = True, detach_qk: bool = True,
                    shapley_attn: bool = False):
    """RelP node attribution: (a^corrupted - a^clean) . grad, a single-point (input x grad)
    attribution where the backward pass uses RelP's relevance rules (see build_relp_fwd_hooks).
    With relp_hooks=False this is exactly input x grad (EAP / 1-step IG) -- used as a self-test."""
    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)   # activation_difference = +corrupted
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        extra = build_relp_fwd_hooks(model, use_qk=detach_qk, shapley_attn=shapley_attn) if relp_hooks else []
        with model.hooks(fwd_hooks=fwd_hooks_clean + extra, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)       # activation_difference -> corrupted - clean
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items
    return scores


def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                              intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', steps=30, 
                              intervention_dataloader: Optional[DataLoader]=None, quiet:bool=False, neuron:bool=False):

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)    
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        if intervention == 'patching':
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

        elif 'mean' in intervention:
            activation_difference += means

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

            activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k/steps
                new_output = alpha * clean + (1 - alpha) * corrupted + activations * 0
                return new_output
            return hook_fn

        total_steps = 0

        nodeslist = [graph.nodes['input']]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append(graph.nodes[f'a{layer}.h0'])
            nodeslist.append(graph.nodes[f'm{layer}'])

        for node in nodeslist:
            for step in range(1, steps+1):
                total_steps += 1
                
                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_tokens, attention_mask=attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                               quiet:bool=False, neuron:bool=False):
    """Gets edge attribution scores using EAP with integrated gradients.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        steps (int, optional): number of IG steps. Defaults to 30.
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)    
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations 
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

        total_steps = 2
        with model.hooks(bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            model.zero_grad()

            logits = model(corrupted_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            model.zero_grad()

    scores /= total_items
    scores /= total_steps

    return scores

allowed_aggregations = {'sum', 'mean'}      
def attribute_node(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                   method: Literal['EAP', 'EAP-IG-inputs', 'EAP-IG-activations', 'exact'], 
                   intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                   aggregation='sum', ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None,
                   quiet:bool=False, neuron:bool=False, optimal_ablation_path: Optional[str]=None):
    # optimal_ablation_path is accepted for compatibility with MIB's run_attribution.py
    # (only the edge-level / 'optimal' ablation path uses it; ignored for these node methods).
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor
    if method == 'EAP':
        scores = get_scores_eap(model, graph, dataloader, metric, intervention=intervention, 
                                intervention_dataloader=intervention_dataloader, quiet=quiet, neuron=neuron)
    elif method == 'EAP-IG-inputs':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for EAP-IG-inputs, but got {intervention}")
        scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet, neuron=neuron)
    elif method == 'EAP-IG-inputs-local':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for EAP-IG-inputs-local, but got {intervention}")
        scores = get_scores_eap_ig_local(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet, neuron=neuron)
    elif method == 'RelP':
        scores = get_scores_relp(model, graph, dataloader, metric, quiet=quiet, neuron=neuron)
    elif method == 'RelP-qkgrad':
        scores = get_scores_relp(model, graph, dataloader, metric, quiet=quiet, neuron=neuron, detach_qk=False)
    elif method == 'AttnRLP':
        scores = get_scores_relp(model, graph, dataloader, metric, quiet=quiet, neuron=neuron, detach_qk=False, shapley_attn=True)
    elif method == 'RelP-norules':
        scores = get_scores_relp(model, graph, dataloader, metric, quiet=quiet, neuron=neuron, relp_hooks=False)
    elif method == 'EAP-IG-activations':
        scores = get_scores_ig_activations(model, graph, dataloader, metric, steps=ig_steps, 
                                           intervention=intervention, intervention_dataloader=intervention_dataloader, 
                                           quiet=quiet, neuron=neuron)
    elif method == 'exact':
        scores = get_scores_exact(model, graph, dataloader, metric, intervention=intervention, 
                                  intervention_dataloader=intervention_dataloader, 
                                  quiet=quiet)
    else:
        raise ValueError(f"integrated_gradients must be in ['EAP', 'EAP-IG-inputs', 'EAP-IG-activations'], but got {method}")


    if aggregation == 'mean':
        scores /= model.cfg.d_model
        
    if neuron:
        graph.neurons_scores[:] = scores.to(graph.scores.device)
    else:
        graph.nodes_scores[:] = scores.to(graph.scores.device)