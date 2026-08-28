# Gradient Descent Proof Audit

Review the proof below. Identify the earliest logical gap, determine whether
the theorem remains true, and provide a complete repair without adding an
assumption stronger than the theorem's stated hypotheses.

## Theorem

Let \(f:\mathbb{R}^d\to\mathbb{R}\) be convex and \(L\)-smooth, and suppose
that a minimizer \(x^*\) exists. Gradient descent uses

\[
x_{t+1}=x_t-\frac{1}{L}\nabla f(x_t).
\]

The claimed rate is

\[
f(x_T)-f(x^*)\leq \frac{L\lVert x_0-x^*\rVert^2}{2T}.
\]

## Submitted proof

Smoothness gives

\[
f(x_{t+1})\leq
f(x_t)-\frac{1}{2L}\lVert\nabla f(x_t)\rVert^2.
\]

Convexity gives

\[
f(x_t)-f(x^*)
\leq \langle\nabla f(x_t),x_t-x^*\rangle
\leq \lVert\nabla f(x_t)\rVert\lVert x_t-x^*\rVert.
\]

Assume throughout the iteration that
\(\lVert x_t-x^*\rVert\leq\lVert x_0-x^*\rVert\). Substituting this bound into
the preceding inequalities and applying the resulting recurrence proves the
claimed rate.

Your audit must distinguish an unstated intermediate claim from a new theorem
assumption, justify every inequality used in the repair, and state explicitly
how the bound telescopes or otherwise yields the final \(1/T\) rate.
