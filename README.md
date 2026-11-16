# MLA
The main part of the MLA is compressed latent vector:

$Q = x W_Q$

$C_{KV} = x \times W_{DKV}$

$K = C_{KV} \times W_{UK}$

$V = C_{KV} \times W_{UV}$

# Matrix Absorption

let's reminder that K is $K = C_{KV} \times W_{UK}$
then: 
	$Q \times K^T = (x_t \times W_Q)(C_{KV} \times W_{UK})^T$

from linear algebra we know that: $(AB)^T = B^T \times A^T$
so:

$(x_t \times W_Q) \times W_{UK}^T \times C_{KV}^T$
$x_t \times (W_Q \times W_{UK}^T) \times C_{KV}^T$
