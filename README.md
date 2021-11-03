# x-transformers-paddle
x-transformers-paddle 2.x version

paddle 2.x版本 https://github.com/lucidrains/x-transformers。

# requirements

- paddlepaddle-gpu==2.2.0-rc0

A concise but fully-featured transformer, complete with a set of promising e**x**perimental features from various papers.

## Usage

Full encoder / decoder

```python
import paddle
from pd_x_transformers import XTransformer

model = XTransformer(
    dim = 512,
    enc_num_tokens = 256,
    enc_depth = 6,
    enc_heads = 8,
    enc_max_seq_len = 1024,
    dec_num_tokens = 256,
    dec_depth = 6,
    dec_heads = 8,
    dec_max_seq_len = 1024,
    tie_token_emb = True      # tie embeddings of encoder and decoder
)

src = paddle.randint(0, 256, (1, 1024))
src_mask = paddle.ones_like(src).astype("bool")
tgt = paddle.randint(0, 256, (1, 1024))
tgt_mask = paddle.ones_like(tgt).astype("bool")

loss = model(src, tgt, src_mask = src_mask, tgt_mask = tgt_mask) # (1, 1024, 512)
loss.backward()
```

Decoder-only (GPT-like)

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 12,
        heads = 8
    )
)

x = paddle.randint(0, 256, (1, 1024))

model(x) # (1, 1024, 20000)
```

GPT3 would be approximately the following (but you wouldn't be able to run it anyways)

```python

gpt3 = TransformerWrapper(
    num_tokens = 50000,
    max_seq_len = 2048,
    attn_layers = Decoder(
        dim = 12288,
        depth = 96,
        heads = 96,
        attn_dim_head = 128
    )
)
```

Encoder-only (BERT-like)

```python
import paddle
from pd_x_transformers import TransformerWrapper, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 12,
        heads = 8
    )
)

x = paddle.randint(0, 256, (1, 1024))
mask = paddle.ones_like(x).astype("bool")

model(x, mask = mask) # (1, 1024, 20000)
```

State of the art image classification

```python
import paddle
from pd_x_transformers import ViTransformerWrapper, Encoder

model = ViTransformerWrapper(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    attn_layers = Encoder(
        dim = 512,
        depth = 6,
        heads = 8,
    )
)

img = paddle.randn((1, 3, 256, 256))
model(img) # (1, 1000)
```

Image -> caption

```python
import paddle
from pd_x_transformers import ViTransformerWrapper, TransformerWrapper, Encoder, Decoder

encoder = ViTransformerWrapper(
    image_size = 256,
    patch_size = 32,
    attn_layers = Encoder(
        dim = 512,
        depth = 6,
        heads = 8
    )
)

decoder = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        cross_attend = True
    )
)

img = paddle.randn((1, 3, 256, 256))
caption = paddle.randint(0, 20000, (1, 1024))

encoded = encoder(img, return_embeddings = True)
decoder(caption, context = encoded) # (1, 1024, 20000)
```

## Dropouts

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    emb_dropout = 0.1,         # dropout after embedding
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        attn_dropout = 0.1,    # dropout post-attention
        ff_dropout = 0.1       # feedforward dropout
    )
)

x = paddle.randint(0, 20000, (1, 1024))
model(x)
```

## Features

### Augmenting Self-attention with Persistent Memory

<img src="./images/all-attention.png" width="500px"></img>

https://arxiv.org/abs/1907.01470

Proposes adding learned memory key / values prior to attention. They were able to remove feedforwards altogether and attain similar performance to the original transformers. I have found that keeping the feedforwards and adding the memory key / values leads to even better performance.

```python
from pd_x_transformers import Decoder, Encoder

enc = Encoder(
    dim = 512,
    depth = 6,
    heads = 8,
    attn_num_mem_kv = 16 # 16 memory key / values
)
```

### Memory Transformers

<img src="./images/memory-transformer.png" width="500px"></img>

https://arxiv.org/abs/2006.11527

Proposes adding learned tokens, akin to CLS tokens, named memory tokens, that is passed through the attention layers alongside the input tokens.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    num_memory_tokens = 20, # 20 memory tokens
    attn_layers = Encoder(
        dim = 512,
        depth = 6,
        heads = 8
    )
)
```

### Transformers Without Tears

<img src="./images/scalenorm.png"></img>

https://arxiv.org/abs/1910.05895

They experiment with alternatives to Layer normalization and found one that is both effective and simpler. Researchers have shared with me this leads to faster convergence.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        use_scalenorm = True # set to true to use for all layers
    )
)
```

### GLU Variants Improve Transformer

<img src="./images/ffglu.png"></img>

https://arxiv.org/abs/2002.05202

Noam Shazeer paper that explores gating in the feedforward, finding that simple gating with GELU leads to significant improvements. This variant also showed up in the latest mT5 architecture. You should always turn this on (I may eventually turn it on by default).

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        ff_glu = True # set to true to use for all feedforwards
    )
)
```

### ReLU²

https://arxiv.org/abs/2109.08668

This paper used neural architecture search and found an activation, Relu Squared, that is both simpler and performs better than GELU, in the autoregressive language model setting. I have confirmed this in my independent experiments. However, if one were using the GLU variant from above, GELU still performs better. Pending further corroboration.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        ff_relu_squared = True
    )
)
```

### Rezero Is All You Need

<img src="./images/rezero.png"></img>

https://arxiv.org/abs/2003.04887

This paper proposes to do away with normalization altogether, and instead gate the output of each branch with a single learned scalar, initialized at zero. They demonstrate convergence for very deep networks, convolution or attention, all without normalization.

I have had good results on usual datasets, but had met trouble with convergence on large datasets (GPT3 sized datasets). However, enough researchers have told me they had positive experiences with this that I decided to include it. If you run into trouble, please use Scalenorm instead.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        use_rezero = True # set to true to use for all layers
    )
)
```

### Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection

<img src="./images/topk-attention.png" width="500px"></img>

https://arxiv.org/abs/1912.11637

This paper proposes an efficient way to sparsify attention by zeroing all dot-product query/key values not within the top k values. The show that this cheap method was as effective as other more expensive operations like sparsemax or entmax15. This technique comes with the cost of an extra hyperparameter (the top k values to keep). The paper recommends a value of `k = 8`

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        attn_sparse_topk = 8 # keep only the top 8 values before attention (softmax)
    )
)
```

Alternatively, if you would like to use `entmax15`, you can also do so with one setting as shown below.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder

# entmax15 TODO paddle version
model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        attn_use_entmax15 = True  # use entmax15 for attention step
    )
)
```

### Talking-Heads Attention

<img src="./images/talking-heads.png" width="500px"></img>

https://arxiv.org/abs/2003.02436

A Noam Shazeer paper that proposes mixing information between heads pre and post attention (softmax). This comes with the cost of extra memory and compute.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        attn_talking_heads = True  # turn on information exchange between attention heads
    )
)
```

### Collaborative Attention

<img src="./images/collaborative-attention.png" width="500px"></img>

https://arxiv.org/abs/2006.16362

Share redundent learned key/query projections accross heads. Collaborative attention reduces the number of parameters but requires slightly more memory and computation. A good compression factor to match the performance of the vanilla multi-head attention is between 0.25 and 0.5.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        attn_collab_heads = True,
        attn_collab_compression = .3,
    )
)
```

### Attention on Attention for Image Captioning

<img src="./images/attention-on-attention.png"></img>

https://arxiv.org/abs/1908.06954

This paper proposes to add a gated linear unit at the end of the attention layer, further gated by the original queries. Although this is not widely used outside of visual question / answering, I suspect it should lead to improvements after seeing the success of the feedforward GLU variant.

Update: After some experimentation, I found this variant actually performs worse, but if it were to be modified to not concatenate the queries before gating, it performs much better. That is what we will be using in this repository.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 6,
        heads = 8,
        attn_on_attn = True  # gate output of attention layer, by queries
    )
)
```

### Intra-attention Gating on Values

<img src="./images/gate_values.png" width="400px"></img>

<a href="https://github.com/deepmind/alphafold">Alphafold2</a> had a peculiar variant of attention where they gate the aggregated values with the input, presumably to have the block have more control over the update.

A quick test shows a small but noticeable improvement, on about the same order as attention on attention.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 6,
        heads = 8,
        attn_gate_values = True  # gate aggregated values with the input
    )
)
```

### Improving Transformer Models by Reordering their Sublayers

<img src="./images/sandwich.png"></img>

<img src="./images/sandwich-2.png"></img>

https://arxiv.org/abs/1911.03864

This paper proposes to break from the normal fixed pattern of alternating attention and feedforwards, but to have blocks of only attention at the beginning followed by blocks of feedforwards at the end. This was further corroborated by a paper by Nvidia that reduces the number of attention layers to be 1/3rd of the feedforwards without loss in performance.

The amount of interleaving is controlled by a "sandwich coefficient", which they found to be optimal at a value of `6`.

You can experiment with this feature as shown below

```python
import paddle
from pd_x_transformers import TransformerWrapper, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 6,
        heads = 8,
        sandwich_coef = 6  # interleave attention and feedforwards with sandwich coefficient of 6
    )
)
```

### Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View

<img src="./images/macaron-1.png"></img>

<img src="./images/macaron-2.png"></img>

https://arxiv.org/abs/1906.02762

The authors propose to view the success of transformers from a dynamical systems point of view, and then proposes an improvement based on mathematics of that POV. Specifically, they propose to place the attention layer in between two feedforward layers. This was adopted by a paper using transformers for speech recognition, the <a href="https://arxiv.org/abs/2005.08100">Conformer</a>.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 6,
        heads = 8,
        macaron = True  # use macaron configuration
    )
)
```

### T5's Simplified Relative Positional Encoding

https://arxiv.org/abs/1910.10683

T5 is one of the most successful encoder / decoder transformer architectures trained to date. They invented a new simplified relative positional encoding based on learned bias values that are added to the attention matrix pre-softmax. This bias is shared and injected into each attention layer. I have decided to include this because it offers a cheap way to have relative positional encoding (superior to absolute positional), and I have read papers that suggest having positional encoding added to each layer (vs only before the first) is beneficial.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        rel_pos_bias = True  # adds relative positional bias to all attention layers, a la T5
    )
)
```

### Position Infused Attention

<img src="./images/pia.png" width="500px"></img>

https://arxiv.org/abs/2005.12872

https://ofir.io/shortformer.pdf

In these two papers, the authors independently figured out a new technique where fixed sinusoidal positional embeddings are injected into the input prior to the queries and keys projection for all layers, leading to "position infused" attention, but leaving the actual tokens (values) uncolored by positional embedding. The Shortformer paper uses this property to cache the tokens for simplified recurrent type of transformer that bested Transformer-XL.

I have tested this, and found that it produces better results than plain absolute positional encoding, even in the absence of recurrence. However, I have found that the T5 relative positional bias (also injected into all layers and has the same properties as PIA) performs even better. So given the option, you should just go with T5's `rel_pos_bias` above.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        position_infused_attn = True  # turns on position infused attention
    )
)
```

### Residual Attention

<img src="./images/residual_attn.png" width="500px"></img>

https://arxiv.org/abs/2012.11747

This paper from Google proposes residualizing the pre-attention scores across all layers. At the cost of no extra parameters, they show improvement on top of regular attention networks. If you turn on this setting, be aware that the best results in the paper used post-normalization, in which case a learning warmup will be needed. The authors also reported that they could use a higher learning rate and get even better gains in the same amount of steps. (In the paper they use `2e-4` vs `1e-4` for vanilla transformer)

```python
import paddle
from pd_x_transformers import TransformerWrapper, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 6,
        heads = 8,
        pre_norm = False,       # in the paper, residual attention had best results with post-layernorm
        residual_attn = True    # add residual attention
    )
)
```

I also tried residualizing cross attention and may have noticed an improvement in convergence. You can try it by setting the `cross_residual_attn` keyword to `True`

```python
import paddle
from pd_x_transformers import XTransformer

model = XTransformer(
    dim = 512,
    enc_num_tokens = 256,
    enc_depth = 6,
    enc_heads = 8,
    enc_max_seq_len = 1024,
    dec_num_tokens = 256,
    dec_depth = 6,
    dec_heads = 8,
    dec_max_seq_len = 1024,
    dec_cross_residual_attn = True     # residualize cross attention
)
```

### Transformer-XL recurrence

You can also do Transformer-XL recurrence, by simply passing in a `max_mem_len` in the `TransformerWrapper` class, and then making sure your `Decoder` has `rel_pos_bias` set to `True`.

Then, you can retrieve the memories at each step with the `return_mems` keyword and pass it to the next iteration.

```python
import paddle
from pd_x_transformers import TransformerWrapper, Decoder

model_xl = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 512,
    max_mem_len = 2048,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        rel_pos_bias = True
    )
)

seg1 = paddle.randint(0, 20000, (1, 512))
seg2 = paddle.randint(0, 20000, (1, 512))
seg3 = paddle.randint(0, 20000, (1, 512))

logits1, mems1  = model_xl(seg1, return_mems = True)
logits2, mems2  = model_xl(seg2, mems = mems1, return_mems = True)
logits3, mems3  = model_xl(seg3, mems = mems2, return_mems = True)
```

### Enhanced recurrence

<img src="./images/enhanced-recurrence.png" width="400px"/>

<a href="https://arxiv.org/abs/2012.15688">This paper</a> proposes a simple technique to enhance



# Reference:

https://github.com/lucidrains/x-transformers