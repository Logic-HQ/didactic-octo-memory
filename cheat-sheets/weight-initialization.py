import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # Dimension for keys and queries (usually d_model is divided by num_heads)
        self.d_k = d_model // num_heads 
        self.num_heads = num_heads
        
        # Weight matrices for Q, K, V. They determine how we transform the input embeddings.
        self.W_q = nn.Linear(d_model, d_model) # For Q
        self.W_k = nn.Linear(d_model, d_model) # For K
        self.W_v = nn.Linear(d_model, d_model) # For V
        
        # Final projection (to combine the results of all heads)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1. Projection (Creating Q, K, V)
        Q = self.W_q(Q) # (Batch, SeqLen, d_model)
        K = self.W_k(K) # (Batch, SeqLen, d_model)
        V = self.W_v(V) # (Batch, SeqLen, d_model)

        # 2. Splitting into heads (Split into heads)
        # Reshape the dimension: (Batch, SeqLen, d_model) -> (Batch, NumHeads, SeqLen, d_k)
        Q = Q.view(batch_size, self.num_heads, -1, self.d_k) # SeqLen * d_k
        K = K.view(batch_size, self.num_heads, -1, self.d_k)
        V = V.view(batch_size, self.num_heads, -1, self.d_k)

        # 3. Calculating Attention (Scaled Dot-Product)
        # Q and K are multiplied along the diagonal, and then the dot product is taken (Dot Product)
        # (Batch, Heads, SeqLen, d_k) @ (Batch, Heads, d_k, SeqLen) -> (Batch, Heads, SeqLen, SeqLen)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. Applying Mask (if present)
        if mask is not None:
            # The mask should be 0 for "forbidden" positions and a very large number (-inf) to block attention
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. Softmax (Obtaining Attention Weights)
        attention_weights = torch.softmax(scores, dim=-1)

        # 6. Weighted Sum of V
        # Multiply weights by V
        context = torch.matmul(attention_weights, V) # (Batch, Heads, SeqLen, d_k)

        # 7. Concatenation of Heads and Final Projection
        # Combine heads back into one vector
        context = context.transpose(1, 2).contiguous() # (Batch, SeqLen, Heads, d_k) -> (Batch, Heads, SeqLen, d_k)
        context = context.view(batch_size, -1, -1, self.d_model) # (Batch, SeqLen, Heads, d_model)

        output = self.W_o(context) # Final projection
        
        return output


# --- Usage Example ---

d_model = 512    # Embedding dimension
num_heads = 8    # Number of attention heads
seq_len = 10     # Sequence length (e.g., 10 words)
batch_size = 4   # Batch size

# Simulation of input data (in a real scenario, this would be Word Embeddings + Positional Encoding)
# Input data must have the shape (Batch Size, Seq Len, D_model)
dummy_input = torch.randn(batch_size, seq_len, d_model)

# Initialize the attention layer
attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

# Since Self-Attention works on one sequence at a time:
# The input data Q, K, V in this case will be the same (dummy_input)
output = attention(
    Q=dummy_input, 
    K=dummy_input, 
    V=dummy_input
)

print(f"Input shape: {dummy_input.shape}")
print(f"Attention output shape: {output.shape}")
# Expected result: (4, 10, 512) - batch and sequence length are preserved

