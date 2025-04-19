
# The Path to Transformer Upgrades: Part 19 — The Second Type of Rotary Position Encoding  

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


[![GitHub stars](https://img.shields.io/github/stars/The-Swarm-Corporation/Legal-Swarm-Template?style=social)](https://github.com/The-Swarm-Corporation/Legal-Swarm-Template)
[![Swarms Framework](https://img.shields.io/badge/Built%20with-Swarms-blue)](https://github.com/kyegomez/swarms)



By **Su Jianlin** | April 18, 2025 | [Original Source](https://kexue.fm/archives/10862)  

Readers who’ve been following the “Path to Transformer Upgrades” series up to this point are likely already familiar with Rotary Position Encoding (RoPE). In short, RoPE applies a rotation transformation to the query (**Q**) and key (**K**) vectors in attention. While it’s a form of **absolute positional encoding**, when combined with the dot-product mechanism in attention, it automatically achieves the effect of **relative positional encoding**.

---

## Can RoPE Be Applied to the Value Vector?

It would seem not—rotating the **value** vector (**V**) would seemingly break the relative position behavior. But that’s not the full story. In this article, we explore applying RoPE to **V**. We refer to this as the **Second Type of Rotary Position Encoding**, or **VO-RoPE** (Value + Output RoPE).

---

## Recap: How RoPE Works

Let's start with the dot-product attention formula:

\[
o_i = \sum_j a_{i,j} v_j, \quad a_{i,j} = \frac{e^{s_{i,j}}}{\sum_j e^{s_{i,j}}}, \quad s_{i,j} = q_i^\top k_j
\]

RoPE applies rotation matrices to the query and key:

\[
q_i \rightarrow R_i q_i, \quad k_j \rightarrow R_j k_j
\]

So the score becomes:

\[
s_{i,j} = (R_i q_i)^\top (R_j k_j) = q_i^\top R_i^\top R_j k_j = q_i^\top R_{j-i} k_j
\]

The final score depends only on **relative position** \((j - i)\). That’s how RoPE achieves relative encoding through an absolute position form.

We also previously showed that the most general solution is \( R_i = O^i \), where \( O \) is an orthogonal matrix. Later work showed that all such orthogonal solutions are essentially equivalent to rotation matrices.

---

## The New Idea: Apply RoPE to V

Now, what if we rotate **v_j** as well?

\[
v_j \rightarrow R_j v_j
\]

Plugging into the output:

\[
o_i = \sum_j a_{i,j} R_j v_j
\]

This introduces explicit **absolute position dependence** into the output. That defeats our goal of a relative positional encoding. But there's a simple trick:

### Apply an Inverse Rotation at Output:

\[
o_i = R_i^\top \left(\sum_j a_{i,j} R_j v_j\right) = \sum_j a_{i,j} R_{j-i} v_j
\]

Now the result depends only on the **relative position**, just like the original RoPE. Since we're applying RoPE on both V and O, we call this **VO-RoPE** (Value + Output RoPE). By comparison, standard RoPE can now be called **QK-RoPE**.

---

## Quick Experiment

We ran experiments on a LLaMA-like model (~1B parameters), comparing several variations:

1. **NoPE**: No positional encoding at all  
2. **QK-RoPE**: Standard rotary on Q and K  
3. **VO-RoPE**: RoPE on V and inverse RoPE on O  
4. **Q/K/V/O-RoPE**: Apply RoPE to just one of Q, K, V, or O  
5. **QKV-RoPE**: Apply RoPE to Q, K, and V  
6. **QKVO-RoPE**: Apply RoPE to all four

### Results (lower loss is better):

| Method       | Loss   |
|--------------|--------|
| QK-RoPE      | 2.712  |
| QKVO-RoPE    | 2.719  |
| K-RoPE       | 2.769  |
| VO-RoPE      | 2.770  |
| QKV-RoPE     | 2.783  |
| NoPE         | 2.795  |
| O-RoPE       | 2.841  |
| Q-RoPE       | 2.851  |
| V-RoPE       | 2.856  |

---

## Thoughts & Takeaways

- VO-RoPE is better than using no position encoding.
- VO-RoPE < QK-RoPE in performance.
- Combining VO-RoPE with QK-RoPE doesn't help.
- Despite limited benefit now, VO-RoPE completes the conceptual framework of RoPE.
- It may be useful in future or niche settings.

---

## Possible Application: MLA (Multi-head Linear Attention)

In MLA (from the article [“The Tug-of-War Between Caching and Effectiveness”](https://kexue.fm/archives/10826)), KV becomes shared, like in MQA:

\[
o_i = \sum_j a_{i,j} c_j, \quad s_{i,j} = q_i^\top c_j
\]

If we apply RoPE on \(c_j\), problems arise:

1. **K and V aren't shared anymore** — this breaks caching.
2. **Applying RoPE to V** makes it absolute again — not what we want.

But with **VO-RoPE**, we can apply RoPE to \(c_j\) and then cancel it with an inverse at the output:

\[
o_i = R_i^\top \sum_j a_{i,j} R_j c_j
\]

It preserves relative encoding **and** KV-sharing. Still, it's not ready for full training yet — just a concept for now.

---

## Connection to Complex-Valued Linear RNNs

VO-RoPE also forms a bridge between attention and **complex linear RNNs**, like LRU or RetNet.

Assume causal attention with exponentially decaying weights:

\[
a_{i,j} = \gamma^{i-j},\quad 0 < \gamma < 1
\]

Then:

\[
o_i = \sum_{j=1}^i \gamma^{i-j} R_{j-i} v_j
\]

Using complex notation: \( R_{j-i} = e^{i\theta(j-i)} \), we get:

\[
o_i = \sum_{j=1}^i (\gamma e^{-i\theta})^{i-j} v_j
\]

This is a **complex exponential decay**, exactly like complex RNNs. So VO-RoPE is the bridge from real to complex linear time decay — again, maybe not practically useful now, but conceptually elegant.

---

## Conclusion

This post asked: _Can RoPE be applied to values?_  
The answer: **Yes — if you cancel it later with an inverse.**  
This is **VO-RoPE**, or the **second kind of RoPE**.

Even if it doesn't bring gains now, it's a **complete and elegant addition** to the RoPE framework and might enable **new tricks in architectures like MLA or complex-valued attention**.

---

## Citation

Su, Jianlin. (2025, April 18). *The Path to Transformer Upgrades: Part 19 – The Second Kind of Rotary Position Encoding*. Retrieved from: https://kexue.fm/archives/10862

