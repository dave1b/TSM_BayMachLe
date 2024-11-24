In Bayesian Machine Learning, **computing the posterior distribution** is often **intractable** when:

1. **Complex Likelihoods or Priors**:
   - If the likelihood $ P(\mathcal{D} | \theta) $ or the prior $ P(\theta) $ is complex (e.g., non-Gaussian, non-conjugate), deriving an analytical form for the posterior becomes impossible.

2. **High-Dimensional Models**:
   - In models with high-dimensional parameter spaces (e.g., deep Bayesian neural networks), the integrations required for marginalizing over parameters are computationally expensive or impossible.

3. **No Closed-Form Solution**:
   - The posterior $ P(\theta | \mathcal{D}) \propto P(\mathcal{D} | \theta) P(\theta) $ may not have a closed-form expression due to the intractability of the evidence $ P(\mathcal{D}) = \int P(\mathcal{D} | \theta) P(\theta) d\theta $.

---

### **Variational Inference (VI)**

Variational Inference is a **deterministic approximation technique** for approximating the posterior. It transforms the problem of inference into an **optimization problem**. Here's how it works:

---

#### **1. Key Idea**
The goal is to approximate the true posterior $ P(\theta | \mathcal{D}) $ with a simpler distribution $ Q_\phi(\theta) $, parameterized by $ \phi $ (e.g., mean and variance of a Gaussian). To find the best $ Q_\phi(\theta) $, we minimize the **KL-divergence** between $ Q_\phi(\theta) $ and $ P(\theta | \mathcal{D}) $:

$
\text{KL}(Q_\phi(\theta) \,||\, P(\theta | \mathcal{D})) = \int Q_\phi(\theta) \log \frac{Q_\phi(\theta)}{P(\theta | \mathcal{D})} d\theta
$

However, directly minimizing this KL-divergence is intractable because $ P(\mathcal{D}) $ (the marginal likelihood or evidence) is hard to compute.

---

#### **2. Evidence Lower Bound (ELBO)**

Instead, we maximize a lower bound on the evidence, called the **Evidence Lower Bound (ELBO)**. The ELBO is derived by rewriting the marginal log-likelihood:

$
\log P(\mathcal{D}) = \mathbb{E}_{Q_\phi(\theta)} \left[ \log \frac{P(\mathcal{D}, \theta)}{Q_\phi(\theta)} \right] + \text{KL}(Q_\phi(\theta) \,||\, P(\theta | \mathcal{D}))
$

Since the KL-divergence is always non-negative, we have:

$
\log P(\mathcal{D}) \geq \mathbb{E}_{Q_\phi(\theta)} \left[ \log \frac{P(\mathcal{D}, \theta)}{Q_\phi(\theta)} \right] = \text{ELBO}
$

To approximate the posterior, we **maximize the ELBO** instead of minimizing the KL-divergence directly:

$
\text{ELBO} = \mathbb{E}_{Q_\phi(\theta)} [\log P(\mathcal{D} | \theta)] - \text{KL}(Q_\phi(\theta) \,||\, P(\theta))
$

---

#### **3. Relation to KL-Divergence**
The ELBO maximization is equivalent to **minimizing the KL-divergence** between $ Q_\phi(\theta) $ and the true posterior $ P(\theta | \mathcal{D}) $:

$
\text{KL}(Q_\phi(\theta) \,||\, P(\theta | \mathcal{D})) = \log P(\mathcal{D}) - \text{ELBO}
$

By maximizing the ELBO, we indirectly minimize the KL-divergence.

---

#### **4. Optimization via Gradient Ascent**

To optimize the ELBO:
1. **Parameterize $ Q_\phi(\theta) $**:
   - Use a distribution family (e.g., Gaussian) and learn its parameters $ \phi $ (e.g., mean and variance).
2. **Monte Carlo Sampling**:
   - Estimate expectations $ \mathbb{E}_{Q_\phi(\theta)}[\cdot] $ using samples from $ Q_\phi(\theta) $.
3. **Gradient Ascent**:
   - Compute the gradient of the ELBO with respect to $ \phi $ and update $ \phi $ using gradient ascent:
     $
     \phi \leftarrow \phi + \eta \nabla_\phi \text{ELBO}
     $

---

#### **5. Gauss-Hermite Quadrature**
When expectations are hard to compute, **Gauss-Hermite quadrature** can approximate integrals efficiently:
1. **Gaussian Approximation**:
   - Assume $ Q_\phi(\theta) $ is Gaussian.
2. **Weighted Sum**:
   - The expectation $ \mathbb{E}_{Q_\phi(\theta)}[f(\theta)] $ is approximated as a weighted sum of $ f(\theta) $ evaluated at specific quadrature points.
   - This reduces variance compared to Monte Carlo methods and is particularly useful for Gaussian posterior approximations.

---

### **Steps in VI with Gauss-Hermite Quadrature**

1. **Initialize**:
   - Choose a variational family $ Q_\phi(\theta) $ (e.g., Gaussian).
2. **Approximate Expectations**:
   - Use quadrature rules to approximate $ \mathbb{E}_{Q_\phi(\theta)}[\cdot] $.
3. **Maximize ELBO**:
   - Perform gradient ascent on the ELBO to update $ \phi $.

---

### **Summary**

- **Intractability** arises when the posterior or evidence cannot be computed analytically.
- **Variational Inference** approximates the posterior by minimizing the KL-divergence between the true posterior and a simpler distribution.
- The **ELBO** is used as a surrogate objective for optimization.
- **Gradient Ascent** or quadrature methods like **Gauss-Hermite quadrature** are used to optimize the ELBO.