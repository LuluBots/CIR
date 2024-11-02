import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from typing import Optional
import string


class LayerNorm(nn.Module):
    """Layer norm used in Transformer layers."""
    
    def __init__(self, dim=1, epsilon=1e-6, use_scale=True, use_bias=True):
        super(LayerNorm, self).__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.use_scale = use_scale
        self.use_bias = use_bias
        
        if self.use_scale:
            self.scale = nn.Parameter(torch.ones(1, dim))  # 调整维度
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, dim))  # 调整维度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normed_x = (x - mean) * (1 / torch.sqrt(var + self.epsilon))

        if self.use_scale:
            normed_x = normed_x * (1 + self.scale)
        if self.use_bias:
            normed_x = normed_x + self.bias

        return normed_x


class Weight(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(Weight, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w = nn.Parameter(torch.empty(input_dim, hidden_dim))

        # 使用 LeCun 正态分布初始化权重
        nn.init.normal_(self.w, mean=0, std=1.0 / (self.input_dim ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w)


class Bias(nn.Module):
    def __init__(self, hidden_dim: int = 0):
        super(Bias, self).__init__()
        self.hidden_dim = hidden_dim
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b


class FFN(nn.Module):
    """Feed-forward network."""
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True, use_relu: bool = True):
        super(FFN, self).__init__()
        self.use_relu = use_relu
        
        # 定义线性层
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.use_relu:
            x = torch.relu(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.layer_norm(x)


class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True, use_relu=True):
        super(FFN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.activation = nn.ReLU() if use_relu else nn.Identity()

    def forward(self, x):
        return self.activation(self.fc(x))


class TransformerFFN(nn.Module):
    def __init__(self, input_dim, output_dim=0, hidden_dim=0, use_bias=True, add_skip_connection=True):
        super(TransformerFFN, self).__init__()
        if output_dim == 0:
            output_dim = input_dim
            
        self.layer_norm = LayerNorm(input_dim)
        self.ffn1 = FFN(input_dim, hidden_dim, use_bias)
        self.ffn2 = FFN(hidden_dim, output_dim, use_bias, use_relu=False)
        self.add_skip_connection = add_skip_connection

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.ffn1(x)
        x = self.ffn2(x)
        if self.add_skip_connection:
            x = x + residual
        return x


class AttentionProjection(nn.Module):
    """Projection (e.g., k) used in self-attention.

    output_proj: Whether it is out projection or not. If False, we use
      "...D,DNH->...NH" for query,key,value projection. Otherwise we use
      "...NH,DNH->...D" for output projection.
    """
    def __init__(self, input_dim: int, num_heads: int, dim_per_head: int, use_bias: bool = True, output_proj: bool = False):
        super(AttentionProjection, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.use_bias = use_bias
        self.output_proj = output_proj

        hd_shape = (num_heads, dim_per_head)
        # pc_shape = (input_dim, *hd_shape) if not output_proj else (*hd_shape, input_dim)
        pc_shape = (input_dim, *hd_shape)

        self.w = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(*pc_shape)))

        if use_bias:
            if output_proj:
                self.b = nn.Parameter(torch.zeros(input_dim))
            else:
                self.b = nn.Parameter(torch.zeros(*hd_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eqn_sym = ''.join(sorted(set(string.ascii_uppercase) - set('DHN')))
        shape = x.shape
        rank = len(shape)

        if self.output_proj:
            assert shape[-2:] == (self.num_heads, self.dim_per_head)
            batch_eqn = eqn_sym[:(rank - 2)]
            eqn = f'{batch_eqn}NH,DNH->{batch_eqn}D'
        else:
            assert shape[-1] == self.input_dim, f'Expecting shape[-1] == p.input_dim, {shape[-1]} != {self.input_dim}'
            batch_eqn = eqn_sym[:(rank - 1)] if rank else '...'
            eqn = f'{batch_eqn}D,DNH->{batch_eqn}NH'
        # print(f"x shape: {x.shape}")
        # print(f"w shape: {self.w.shape}")
        # print(f"Equation: {eqn}")

        ret = torch.einsum(eqn, x, self.w)
        if self.use_bias:
            ret += self.b

        return ret



class PerDimScale(nn.Module):
    def __init__(self, dim: int):
        super(PerDimScale, self).__init__()
        self.dim = dim
        # 初始化可学习参数
        self.per_dim_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.dim
        r_softplus_0 = 1.442695041
        # 计算缩放因子
        scale = r_softplus_0 / (self.dim ** 0.5)  # 使用平方根
        scale *= torch.nn.functional.softplus(self.per_dim_scale)
        return x * scale


class DotProductAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=1, use_bias=True, dim_per_head=0, use_per_dim_scale=False):
        super(DotProductAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.dim_per_head = dim_per_head or (hidden_dim // num_heads)
        assert (self.dim_per_head * num_heads == hidden_dim), f"{self.dim_per_head} * {num_heads} != {hidden_dim}"

        self.key = AttentionProjection(input_dim, num_heads, self.dim_per_head, use_bias)
        self.query = AttentionProjection(input_dim, num_heads, self.dim_per_head, use_bias)
        self.value = AttentionProjection(input_dim, num_heads, self.dim_per_head, use_bias)

        if use_per_dim_scale:
            self.per_dim_scale = nn.Parameter(torch.ones(num_heads, self.dim_per_head))

        self.post = AttentionProjection(input_dim, num_heads, self.dim_per_head, use_bias, output_proj=True)
        

    def _dot_atten(self, query, key, value):
        if hasattr(self, 'per_dim_scale'):
            query = query * self.per_dim_scale.view(1, self.num_heads, 1, self.dim_per_head)
        else:
            query *= self.dim_per_head ** -0.5

        logits = torch.einsum('BTNH,BSNH->BNTS', query, key)
        cap = 50.0
        logits = cap * torch.tanh(logits / cap)
        probs = F.softmax(logits, dim=-1)
        encoded = torch.einsum('BNTS,BSNH->BTNH', probs, value)

        return encoded, probs

    def forward(self, q_vector, k_vector, v_vector, atten_mask=None):
        query_proj = self.query(q_vector)
        key_proj = self.key(k_vector)
        value_proj = self.value(v_vector)
        encoded, atten_probs = self._dot_atten(query_proj, key_proj, value_proj)
        encoded = self.post(encoded)
        return encoded, atten_probs


class Transformer(nn.Module):
    """Transformer layer used in multimodal encoder."""
    def __init__(self, num_heads: int, input_dim: int = 0, hidden_dim: int = 0, 
                 output_dim: int = 0, use_bias: bool = True, 
                 add_skip_connection: bool = True, use_per_dim_scale: bool = False):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim != 0 else input_dim
        self.use_bias = use_bias
        self.add_skip_connection = add_skip_connection
        self.use_per_dim_scale = use_per_dim_scale

        # Initialize layers
        self.ff_layer = TransformerFFN(
            self.input_dim,
            self.output_dim,
            self.hidden_dim,
            self.use_bias,
            self.add_skip_connection
        )
        
        attn_hidden_dim = self.input_dim
        self.self_attention = DotProductAttention(
            input_dim=self.input_dim,
            hidden_dim=attn_hidden_dim,
            num_heads=self.num_heads,
            use_bias=self.use_bias,
            use_per_dim_scale=self.use_per_dim_scale
        )
        self.layer_norm = LayerNorm(dim=self.input_dim)

    def forward(self, x: torch.Tensor, attn_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("Hi.")
        x_normalized = self.layer_norm(x)
        atten_output, atten_probs = self.self_attention(
            x_normalized,
            x_normalized,
            x_normalized,
            atten_mask=attn_mask
        )
        if self.add_skip_connection:
            atten_output = atten_output + x
        output = self.ff_layer(atten_output)
        # print("Bye.")

        return output, atten_probs


class StackedTransformer(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, input_dim: int, 
                 hidden_dim: int, use_bias: bool = True, 
                 add_skip_connection: bool = True, use_per_dim_scale: bool = False):
        super(StackedTransformer, self).__init__()
        assert num_layers > 0, "Number of layers must be greater than 0"
        assert input_dim > 0, "Input dimension must be greater than 0"
        assert hidden_dim > 0, "Hidden dimension must be greater than 0"
        assert num_heads > 0, "Number of heads must be greater than 0"
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.add_skip_connection = add_skip_connection
        self.use_per_dim_scale = use_per_dim_scale

        # Initialize the transformer layers
        self.layers = nn.ModuleList([
            Transformer(num_heads=self.num_heads,
                        input_dim=self.input_dim,
                        hidden_dim=self.hidden_dim,
                        output_dim=self.input_dim,  # Assuming output_dim equals input_dim
                        use_bias=self.use_bias,
                        add_skip_connection=self.add_skip_connection,
                        use_per_dim_scale=self.use_per_dim_scale)
            for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x, attn_mask)
        return x


class AttenTokenPoolingLayer(nn.Module):
    def __init__(self, input_dim: int = 0, query_dim: Optional[int] = None,
                 hidden_dim: int = 0, num_heads: int = 1, 
                 num_query_tokens: int = 1, use_bias: bool = True, 
                 use_per_dim_scale: bool = True):
        super(AttenTokenPoolingLayer, self).__init__()
        assert input_dim > 0, 'input_dim must be positive'
        
        self.input_dim = input_dim
        self.query_dim = query_dim if query_dim is not None else input_dim
        self.hidden_dim = hidden_dim if hidden_dim > 0 else 4 * input_dim
        self.num_heads = num_heads
        self.num_query_tokens = num_query_tokens
        self.use_bias = use_bias
        self.use_per_dim_scale = use_per_dim_scale

        # Create sub-modules
        self.pool_attn = DotProductAttention(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            use_bias=self.use_bias,
            use_per_dim_scale=self.use_per_dim_scale
        )
        self.pool_attn_ln = LayerNorm(dim=self.query_dim)

        # Initialize pooling attention query
        self.pooling_attn_query = nn.Parameter(
            torch.empty(self.num_query_tokens, self.query_dim)
        )
        nn.init.xavier_uniform_(self.pooling_attn_query)  # Initialize with Xavier

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """Pooling layer.

        Args:
          embeds: (batch_size, seq_len, input_dim)
        Returns:
          pooled_output: (batch_size, query_dim)
        """
        batch_size = embeds.size(0)
        query = self.pooling_attn_query.unsqueeze(0).expand(batch_size, -1, -1)
        key = embeds
        pooled_output, _ = self.pool_attn(query, key, embeds)
        pooled_output = self.pool_attn_ln(pooled_output)

        return pooled_output
