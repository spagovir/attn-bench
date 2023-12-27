# %%
try:
    from flash_attention_jax import flash_attention
    import jax.numpy as jnp
    is_jax = True
except ImportError:
    import torch as t
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    is_jax = False
# %%
# %%
context_ranges = range(11,17)
head_dims = [64,128,256]
num_heads = [6,8,12,16,20,25]
b_size = [1,5,10]
from time import time
import numpy as np

save_path = "./bench.csv"

# %%
# %%
generator = np.random.default_rng()
# # %%
# b = 5
# n = 6
# h = 32
# s = 1024
# qkv = generator.normal(0.0, 1.0, (b,s,3,n,h)).astype(np.float16, casting="same_kind")
# # %%
# import torch as t
# start_t = time()
# res = flash_attn_qkvpacked_func(t.from_numpy(qkv).to('cuda'))
# end_t = time()
# # %%
# import jax.numpy as jnp
# # %%
# q = jnp.asarray(qkv[:,:,0,:,:])
# start_t = time()

# # %%
# warm up the jax JIT
# if is_jax:
    # qkv = generator.normal(0.0,1.0,(1,1,3,1,1)).astype(np.float16, casting ="same_kind")
    # q = jnp.asarray(qkv[:,:,0,:,:])
    # k = jnp.asarray(qkv[:,:,1,:,:])
    # v = jnp.asarray(qkv[:,:,2,:,:])
    # mask = jnp.ones((1,1), np.int8)
    # ret = flash_attention(q,k,v, mask).block_until_ready()
# # %%
# %%
for ls in context_ranges:
    for h in head_dims:
        for n in num_heads:
            for b in b_size: 
                s = 2**ls
                qkv = generator.normal(0.0,1.0, (b,2**ls, 3, n, h)).astype(np.float16, casting="same_kind")
                if is_jax:
                    q = jnp.asarray(qkv[:,:,0,:,:])
                    k = jnp.asarray(qkv[:,:,1,:,:])
                    v = jnp.asarray(qkv[:,:,2,:,:])
                    mask = jnp.ones((b,2**ls), np.int8)
                    #warmup jax
                    flash_attention(q,k,v,mask).block_until_ready()
                    start_t = time()
                    flash_attention(q,k,v,mask).block_until_ready()
                    end_t = time()
                    print(f"{s=}, {h=}, {n=}, {b=}, {end_t-start_t}s")
                else:
                    qkv = t.from_numpy(qkv).to('cuda')
                    start_t = time()
                    flash_attn_qkvpacked_func(qkv)
                    end_t = time()
                    print(f"{s=}, {h=}, {n=}, {b=}, {end_t-start_t}s")
                    
# %%
