# EfficientUrbanDynamicPathPlanning


## 1. Version Update

- v1.0: initial version.
- v1.1: eliminate $O(n)$ hot-path ops in training loop (env/buffer/trainer).


## 2. Experiment Logs

$$s_t = (v^t,v_g,f_t), \quad f_t \subseteq [0.01,1.0]^n$$

$$R(s,a,s') = -c(s,s') + e \cdot \mathbb{I}(s'=s_g)$$

$$c(s,s') = \frac{l(s,s')}{v_0 * \beta}, \quad \beta \in [0.01,1.0]$$

$$R'(s,a,s') = R_0(s,a,s') + \gamma \Phi_d(s') - \Phi_d(s)$$

$$\Phi_d(s) = - \zeta \cdot d(s,s_g)$$

$$R_v(s,a,s') = R_0(s,a,s') + \gamma \Phi_v(s') - \Phi_v(s)$$

$$\Phi_v(s) = \eta V(s)$$

Main parameters: $e, \beta, \zeta, \eta$

Performance indicators: Success Rate (SR), Mean Gap (GAP), 