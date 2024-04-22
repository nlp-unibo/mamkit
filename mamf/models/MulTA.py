import torch
from core import MAMFBase
from ..modules.transformer_modules import PositionwiseFeedForward, PositionalEncoding

class MulTA_CrossAttentionBlock(nn.Module):
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
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.mh_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.num_heads, dropout=self.dropout_prob, batch_first=True)
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

class MAMFMulTA(MAMFBase):
    """
    Class for the unaligned multimodal model
    """
    def __init__(self, embedding_dim, d_ffn, n_blocks, head, dropout_prob=0.1, hidden_state_index=8):
        """
        Args:
            embedding_dim: dimension of the embedding
            d_ffn: dimension of the feed forward layer
            n_blocks: number of blocks to use
            head: head to use
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.d_ffn = d_ffn
        self.n_blocks = n_blocks
        self.head = head
        self.dropout_prob = dropout_prob
        self.text_crossmodal_blocks = nn.ModuleList([
            MulTA_CrossAttentionBlock(self.embedding_dim, self.d_ffn, dropout_prob=self.dropout_prob) for _ in range(self.n_blocks)
        ])
        self.audio_crossmodal_blocks = nn.ModuleList([
            MulTA_CrossAttentionBlock(self.embedding_dim, self.d_ffn,  dropout_prob=self.dropout_prob) for _ in range(self.n_blocks)
        ])
        self.pos_encoder = PositionalEncoding(embedding_dim, dual_modality=False)
        self.hidden_state_index = hidden_state_index
    
    def forward(self, texts, audio_features, audio_attentions):
        """
        Forward pass of the model
        Args:
            texts: texts to use
            audio_features: audio features to use
            audio_attentions: audio attentions to use
        """
        tokenizer_output = tokenizer(texts, return_tensors='pt', padding=True, truncation=False).to(device)
        embedder_output = embedder(**tokenizer_output, output_hidden_states=True)
        if self.hidden_state_index == -1:
            text_features = embedder_output['last_hidden_state']
        else:
            assert self.hidden_state_index < len(embedder_output['hidden_states']), f'hidden_state_index must be between 0 and {len(embedder_output["hidden_states"])}'
            text_features = embedder_output['hidden_states'][self.hidden_state_index]
        text_features = self.pos_encoder(text_features)
        text_attentions = tokenizer_output.attention_mask
        
        audio_features = self.pos_encoder(audio_features)
        
        # cross modal attention blocks for text
        # using audio features as key and value and text features as query
        text_crossmodal_out = text_features
        for cm_block in self.text_crossmodal_blocks:
            text_crossmodal_out = cm_block(text_crossmodal_out, audio_features, audio_attentions)
        
        # cross modal attention blocks for audio
        # using text features as key and value and audio features as query
        audio_crossmodal_out = audio_features
        for cm_block in self.audio_crossmodal_blocks:
            audio_crossmodal_out = cm_block(audio_crossmodal_out, text_features, text_attentions)

        # pooling of transformer output
        text_crossmodal_out_mean = torch.mean(text_crossmodal_out, dim=1)
        audio_crossmodal_out_mean = torch.mean(audio_crossmodal_out, dim=1)
        
        # concatenate text and audio features
        text_audio = torch.cat((text_crossmodal_out_mean, audio_crossmodal_out_mean), dim=-1)
        
        return self.head(text_audio)