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

## Exercise 3: Time Travel via Gradients (grad)
For the third exercise, I constructed a complete gradient descent optimization loop running for exactly 20 iterations to find the ideal initial velocity for a projectile. Starting from an initial guess of $v_{\text{initial}} = 10.0$, the script evaluated the mean squared error loss against a strict target distance of 150.0 meters after 5 seconds of flight. By extracting the exact analytical gradient in each step via `jax.grad` and applying a parameter update with a learning rate of 0.01, the simulation successfully guided the velocity to its optimal value of exactly 30.0 meters per second.

### Analytical Gradients (jax.grad) vs. Finite Differences
The fundamental difference is that `jax.grad` utilizes backward-mode automatic differentiation to compute the exact analytical slope by evaluating the exact mathematical derivatives along the graph execution path in a single backward pass. In contrast, approximating the slope via finite differences using the formula $\frac{f(x+h)-f(x)}{h}$ requires running the full forward simulation multiple times with a tiny numerical perturbation $h$, making it highly vulnerable to severe floating-point roundoff errors and causing it to scale poorly as the number of optimization parameters increases.