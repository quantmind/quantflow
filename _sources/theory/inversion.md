---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# From Characteristic Function to PDF

+++

One uses the inverse Fourier transform formula to obtain the probability density function (PDF) from a characteristic function.

\begin{equation}
  f(x) = \frac{1}{2\pi}\int_{-\infty}^\infty e^{-iux} \Phi_x\left(u\right) du = \frac{1}{\pi} {\mathcal R} \int_0^\infty e^{-iux} \Phi_u du
\end{equation}

The last equivalence is because the PDF is real-valued.

## Discretization

The PDF integration can be approximated as:

\begin{align}
u_m &= \delta_u m \\ 
f(x) &\approx \frac{1}{\pi}\sum_{m=0}^{N-1} h_m e^{-i u_m x} \Phi_x\left(u_m\right) \delta_u
\end{align}

* $\delta_u$ is the discretization in the frequency domain. It must be small enough to provide good accuracy of the integral.
* $N$ is the number of discretization points and must be large enough so that the characteristic function is virtually 0 for $u_{N-1}=\delta_u N$.
* $h_m$ is given by the integration methodology, either trapezoidal or Simpson rule (the library support both, with trapezoidal as default).

For full details, follow {cite:p}`carr_madan`, {cite:p}`saez`.

One could use the inverse Fourier transform to solve the integral. However, it has $O(N^2)$ time complexity.
One alternative, implemented in the library, is using the Fast Fourier Transform (FFT), which has $O(N \log N)$ time complexity.
Another more flexible alternative is the Fractional FFT as described in {cite:p}`frft`. This is the methodology used by default in the library.

+++

## FFT Integration

FFT is an efficient algorithm for computing discrete Fourier coefficients $d$ from $f$. Gievn an event number $N=2^n$, these are given by

\begin{equation}
d_j = \frac{1}{N}\sum_{m=0}^{N-1} f_m e^{-jm\frac{2\pi}{N} i}\ \ j=0, 1, \dots, N-1
\end{equation}

Using this formula, the discretization above can be rewritten as

\begin{align}
x_j &= -b + \delta_x j \\
\zeta &= \delta_u \delta_x \\
f_m &= h_m \frac{N}{\pi} e^{i u_m b} \Phi_x\left(u_m\right) \delta_u\\
f(x_j) &\approx \frac{1}{N} \sum_{m=0}^{N-1} f_m e^{-j m \zeta i}
\end{align}

The parameter $b$ controls the range of the random variable $x$. The FFT requires that

\begin{equation}
\zeta = \frac{2\pi}{N}
\end{equation}

which means $\delta_u$ and $\delta_x$ cannot be chosen indipendently.

As an example, let us invert the characteristic function of the Weiber process, which yields the standard distribution.

```{code-cell}
from quantflow.sp.weiner import WeinerProcess
p = WeinerProcess(sigma=0.5)
m = p.marginal(0.2)
m.std()
```

```{code-cell}
from quantflow.utils import plot

plot.plot_characteristic(m)
```

```{code-cell}
from quantflow.utils import plot
import numpy as np
plot.plot_marginal_pdf(m, 128, use_fft=True, max_frequency=20)
```

```{code-cell}
plot.plot_marginal_pdf(m, 128*8, use_fft=True, max_frequency=8*20)
```

**Note** the amount of unnecessary discretization points in the frequency domain (the characteristic function is zero after 15 or so). However the space domain is poorly represented because of the FFT constraints (we have a relatively small number of points where it matters, around zero).

\begin{equation}
\delta_x = \frac{2 \pi}{N} \delta_u
\end{equation}

+++

## FRFT
Compared to the FFT, this method relaxes the constraint $\zeta=2\pi/N$ so that frequency domain and space domains can be discretized independently. We use the methodology from {cite:p}`frft`

\begin{align}
y &= \left(\left[e^{-i j^2 \zeta/2}\right]_{j=0}^{N-1}, \left[0\right]_{j=0}^{N-1}\right) \\
z &= \left(\left[e^{i j^2 \zeta/2}\right]_{j=0}^{N-1}, \left[e^{i\left(N-j\right)^2 \zeta/2}\right]_{j=0}^{N-1}\right)
\end{align}

We can now reduce the number of points needed for the discretization and achieve higher accuracy by properly selecting the domain discretization independently.

```{code-cell}
plot.plot_marginal_pdf(m, 128)
```

Since one N-pointFRFTt will invoke three 2N-pointFFTt procedures, the number of operations will be approximately $6N\log{N}$ compared to $N\log{N}$ for the FFT. However, we can use fewer points as demonstrated and be more robust in delivering results.

The FRFT is used as the default transforms across the library, the FFT can be used by passing `use_fft` to the transform functions, but it is not advised.

+++

## Additional References


* [Fourier Transfrom and Characteristic Functions](https://faculty.baruch.cuny.edu/lwu/890/ADP_Transform.pdf) - useful but lots of typos
