# MLA
The main part of the MLA is compressed latent vector:
$Q = x W_Q$

$C_{KV} = x \times W_{DKV}$

$K = C_{KV} \times W_{UK}$

$V = C_{KV} \times W_{UV}$

## Matrix Absorption

$$
Q = x W_Q
$$

$$
C_{KV} = x W_{DKV}
$$

$$
K = C_{KV} W_{UK}
$$

$$
V = C_{KV} W_{UV}
$$

Let's recall that:

$$
K = C_{KV} W_{UK}
$$

Then:

$$
Q K^T = (x_t W_Q)(C_{KV} W_{UK})^T
$$

From linear algebra, we know that:

$$
(AB)^T = B^T A^T
$$

Therefore:

$$
Q K^T = (x_t W_Q) W_{UK}^T C_{KV}^T
$$

Rearranging the terms:

$$
\boxed{
Q K^T = x_t (W_Q W_{UK}^T) C_{KV}^T
}
$$

## DeepSeekMoE

$$
h'_t = u_t + \sum_{i=1}^{N_s} \operatorname{FFN}^{(s)}_i(u_t) + \sum_{i=1}^{N_r} g_{i,t} \operatorname{FFN}^{(r)}_i(u_t)
$$

$$
s_{i,t} = \operatorname{Softmax}_i\left(u_t^T e_i\right), \qquad
g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{TopK}\left(\{s_{j,t}\}_{j=1}^{N_r}, K_r\right) \\ 0, & \text{otherwise} \end{cases}
$$

В DeepSeek-V2: $N_r = 160$, $K_r = 6$, $N_s = 2$, `moe_intermediate_size` $= 1536$ (против 12288 у dense FFN — fine-grained segmentation).

Device-limited routing: эксперты разбиты на $D = 8$ групп по 20; score группы — максимум $s_{i,t}$ в ней; выбираются top-3 группы, и только среди их экспертов берётся top-6. Веса не ре-нормируются, а умножаются на `routed_scaling_factor` $= 16$.

Balance losses:

$$
\mathcal{L}_{ExpBal} = \alpha_1 \sum_{i=1}^{N_r} f_i P_i, \qquad
f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathbb{1}(\text{token } t \text{ selects expert } i), \qquad
P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t}
$$

$$
\mathcal{L}_{DevBal} = \alpha_2 \sum_{i=1}^{D} f'_i P'_i, \qquad
f'_i = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j, \qquad
P'_i = \sum_{j \in \mathcal{E}_i} P_j
$$

$\alpha_1 = \alpha_2 = 0.001$.
