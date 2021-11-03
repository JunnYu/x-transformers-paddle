# x-transformers-paddle
x-transformers-paddle 2.x version

paddle 2.x版本 https://github.com/lucidrains/x-transformers。

# requirements

- paddlepaddle-gpu==2.2.0-rc0

# Example

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
o = decoder(caption, context = encoded) # (1, 1024, 20000)

print(o.shape)
# [1, 1024, 20000]
```

