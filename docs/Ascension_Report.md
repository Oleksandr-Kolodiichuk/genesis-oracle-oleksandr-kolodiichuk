# Problem Set 4: Project Genesis – The Silicon Ascension

## Exercise 1: The Legacy Chokehold (Sequential Simulation)
For Exercise 1, I established a baseline using standard sequential computing via NumPy on a standard CPU. The simulation processed 100,000 independent damped harmonic oscillators over 1,000 discrete time steps using an explicit Euler integration loop. Due to the inherent overhead of Python loops and sequential array updates, the execution was slow and bound heavily by CPU performance.

**Baseline Performance:**
- **Total Execution Time (Legacy NumPy):** 3.2084 seconds

## Exercise 2: The Tensor Multiverse (vmap & jit)
In this exercise, I migrated the physical model to JAX to take advantage of hardware acceleration (GPU/TPU) and functional purity. I transformed the scalar physical logic into a vectorized batch-processing engine using `jax.vmap` and fused the outer loop into the hardware accelerator using `@jax.jit` combined with `jax.lax.fori_loop`.

### Performance Telemetry & Speedup Factor
- **JAX Run 1 (Tracing + Compilation):** 1.4883 seconds
- **JAX Run 2 (Pure Compiled Execution):** 0.001607 seconds

**Speedup Factor Calculation:**
$$\text{Speedup Factor} = \frac{\text{Legacy Time}}{\text{JAX Run 2 Time}} = \frac{3.2084}{0.001607} \approx 1996.51\times$$

The JAX implementation achieved a phenomenal speedup factor of approximately **1,996.5x** over the legacy sequential implementation.

### The JIT Compilation Phenomenon
The first run of a JIT-compiled function is always significantly slower than the second run because JAX must perform an initial "tracing" phase to build an abstract computation graph, which the XLA compiler then translates into optimized machine code for the hardware accelerator. During the second run, JAX bypasses the Python interpreter and the compilation phase entirely, immediately executing the cached native machine code at maximum hardware speeds.