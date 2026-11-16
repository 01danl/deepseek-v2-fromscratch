# MLA
The main part of the MLA is compressed latent vector:

## Matrix Absorption

\[
Q = x W_Q
\]

\[
C_{KV} = x W_{DKV}
\]

\[
K = C_{KV} W_{UK}
\]

\[
V = C_{KV} W_{UV}
\]

Let's recall that:

\[
K = C_{KV} W_{UK}
\]

Then:

\[
Q K^T = (x_t W_Q)(C_{KV} W_{UK})^T
\]

From linear algebra, we know that:

\[
(AB)^T = B^T A^T
\]

Therefore:

\[
Q K^T = (x_t W_Q) W_{UK}^T C_{KV}^T
\]

Rearranging the terms:

\[
\boxed{
Q K^T = x_t (W_Q W_{UK}^T) C_{KV}^T
}
\]
