# %%
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
# %%
context_ranges = range(11,17)
head_dims = [64,128,256]
num_heads = [6,8,12,16,20,25]
b_size = [1,5,10]
from time import time
import numpy as np

save_path = "./bench.csv"

# %%
from flash_attention_jax import flash_attention
import numpy as np
# %%
generator = np.random.default_rng()
# %%
b = 5
n = 6
h = 32
s = 1024
qkv = generator.normal(0.0, 1.0, (b,s,3,n,h)).astype(np.float16, casting="same_kind")
# %%
import torch as t
start_t = time()
res = flash_attn_qkvpacked_func(t.from_numpy(qkv).to('cuda'))
end_t = time()
# %%
import jax.numpy as jnp
# %%
q = jnp.asarray(qkv[:,:,0,:,:])
start_t = time()

# %%
