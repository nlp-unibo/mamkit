import math

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """
        Positional Encoding for Transformer
    """

    def __init__(self, d_model: int, dual_modality=False, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: dimension of model
            dual_modality: when True, add a sequence of 0s or 1s depending on the modality
            dropout: dropout rate
            max_len: max length of sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dual_modality = dual_modality
        self.pe = self.pe.to(device)

    def forward(self, x, is_first=True):
        """
        Args:
            x: input tensor (bs, sqlen, emb)
            is_first: True if the first modality, False if the second modality
        """
        if self.dual_modality:
            modality = torch.ones((x.shape[0], x.shape[1], 4), dtype=torch.float32).to(device) * (0 if is_first else 1)
            x = x + self.pe[:, :x.size(1)]
            x = self.dropout(x)
            return torch.cat((x, modality), dim=-1)
        else:
            # x = (bs, sqlen, emb)  pe = (1, sqlen, emb)
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)


class CustomScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(CustomScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, text_mask, audio_mask, e=1e-12):
        """
        Args:
            q: query (decoder)
            k: key (encoder)
            v: value (encoder)
            text_mask: mask for text sequence
            audio_mask: mask for audio sequence
            e: epsilon value for masking
        """
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        ## score dimension: (batch, n_heads, length, length)

        # 2. apply masking (opt)
        padding_mask = torch.cat((text_mask, audio_mask), dim=1).unsqueeze(1)
        ## padding_mask is now (batch, 1, seq_length)
        score = score.masked_fill(padding_mask.unsqueeze(-1) == 0,
                                  -10_000)  # padding_mask applied = (batch, 1, seq_length, 1)
        score = score.masked_fill(padding_mask.unsqueeze(1) == 0,
                                  -10_000)  # padding_mask applied = (batch, 1, 1, seq_length)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multipy score with attention coefficients
        text_lengths = torch.sum(text_mask, dim=1)
        audio_lengths = torch.sum(audio_mask, dim=1)
        ## text_length and audio_length dimension = (batch,)

        total_lengths = text_lengths + audio_lengths
        ## total_lengths dimension = (batch,)

        text_coefficients = (total_lengths / (2 * text_lengths)).unsqueeze(-1)
        audio_coefficients = (total_lengths / (2 * audio_lengths)).unsqueeze(-1)
        ## text_coefficients and audio_coefficients dimension = (batch, 1)

        text_weights = text_mask * text_coefficients
        audio_weights = audio_mask * audio_coefficients
        ## text_weights dimension = (batch, text_sequence_length)
        ## audio_weights dimension = (batch, audio_sequence_length)

        attention_coefficients = torch.cat((text_weights, audio_weights), dim=1).unsqueeze(1).unsqueeze(1)
        ## attention_coefficients dimension = (batch, 1, 1, total_sequence_length)
        score = score * attention_coefficients

        # 5. multiply with Value
        v = score @ v

        return v, score


class CustomMultiHeadAttention(nn.Module):
    """
    Multi Head Attention Class for Transformer
    """

    def __init__(self, d_model, n_head):
        """
        Args:
            d_model: dimension of model
            n_head: number of heads
        """
        super(CustomMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = CustomScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, text_mask, audio_mask):
        """
        Args:
            q: query (decoder)
            k: key (encoder)
            v: value (encoder)
            text_mask: mask for text sequence
            audio_mask: mask for audio sequence
        """
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, text_mask, audio_mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        Args:
            tensor: [batch_size, length, d_model]
        """

        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)
        Args:
            tensor: [batch_size, head, length, d_tensor]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class LayerNorm(nn.Module):
    """
    Layer Normalization Class
    """

    def __init__(self, d_model, eps=1e-12):
        """
        Args:
            d_model: dimension of model
            eps: epsilon value for masking
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: input tensor
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Layer
    """

    def __init__(self, d_model, hidden, drop_prob=0.1):
        """
        Args:
            d_model: dimension of model
            hidden: dimension of hidden layer
            drop_prob: dropout rate
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        Args:
            x: input tensor
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class CustomEncoderLayer(nn.Module):
    """
    Encoder Layer Class
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
        Args:
            d_model: dimension of model
            ffn_hidden: dimension of hidden layer
            n_head: number of heads
            drop_prob: dropout rate
        """
        super(CustomEncoderLayer, self).__init__()
        self.attention = CustomMultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, text_mask, audio_mask):
        """
        Args:
            x: input tensor
            text_mask: mask for text sequence
            audio_mask: mask for audio sequence
        """
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, text_mask=text_mask, audio_mask=audio_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class CustomEncoder(nn.Module):
    """
    Encoder Class
    """

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        """
        Args:
            d_model: dimension of model
            ffn_hidden: dimension of hidden layer
            n_head: number of heads
            n_layers: number of layers
            drop_prob: dropout rate
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [CustomEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob) for _ in
             range(n_layers)])

    def forward(self, embedding, text_mask, audio_mask):
        """
        Args:
            embedding: input tensor
            text_mask: mask for text sequence
            audio_mask: mask for audio sequence
        """
        x = embedding
        for layer in self.layers:
            x = layer(x, text_mask, audio_mask)

        return x


class MulTA_CrossAttentionBlock(torch.nn.Module):
    """
    Class for the cross modal attention block
    """

    def __init__(self, embedding_dim, d_ffn, num_heads=4, dropout_prob=0.1):
        """
        Args:
            embedding_dim: dimension of the embedding
            d_ffn: dimension of the feed forward layer
            num_heads: number of heads to use
            dropout_prob: dropout to use
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.d_ffn = d_ffn
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.mh_attention = torch.nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.num_heads,
                                                        dropout=self.dropout_prob, batch_first=True)
        self.pointwise_ff = PositionwiseFeedForward(d_model=self.embedding_dim, hidden=self.d_ffn)

    def forward(self, elem_a, elem_b, attn_mask):
        """
        Forward pass of the model
        Args:
            elem_a: elements of the modality A
            elem_b: elements of the modality B
            attn_mask: attention mask to use
        """
        elem_a = self.layer_norm(elem_a)
        elem_b = self.layer_norm(elem_b)
        attn_mask = attn_mask.to(torch.float32)

        # cross modal attention with elem_a as query and elem_b as key and value
        mh_out, _ = self.mh_attention(elem_a, elem_b, elem_b, key_padding_mask=attn_mask, need_weights=False)
        # residual connection
        add_out = mh_out + elem_a

        add_out_norm = self.layer_norm(add_out)
        out_ffn = self.pointwise_ff(add_out_norm)
        out = out_ffn + add_out
        return out
