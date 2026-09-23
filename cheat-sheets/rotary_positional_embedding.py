import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 1. Creating Frequencies (Theta)
        # This determines how quickly the vectors rotate at different positions.
        # We use a formula similar to the one in the original paper.
        self.freqs = self._calculate_freqs(d_model)

    def _calculate_freqs(self, d_model):
        # Create a set of frequencies (rotation coefficients) for each dimension
        # We divide d_model by 2, as we are working with sin/cos pairs.
        dim = d_model // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / d_model))
        return freqs

    def forward(self, x):
        # x has shape: (Batch, SeqLen, D_model)
        batch_size, seq_len, _ = x.shape
        
        # 1. Creating Positional Vector (Positional Indices)
        # [0, 1, 2, ..., SeqLen-1] -> (SeqLen, 1)
        position = torch.arange(seq_len, device=x.device)
        
        # 2. Generating Rotation Multipliers
        # We only use even indices to create sin and cos pairs
        freqs = self.freqs[:self.d_model] # Use only the necessary frequencies

        # Create position vectors that will be used for shifting (m)
        # shape: (SeqLen, D_model/2)
        position_indices = position.unsqueeze(1) * freqs 
        
        # Split x into pairs (to work with sin and cos separately)
        x_sin = torch.sin(position_indices)
        x_cos = torch.cos(position_indices)

        # 3. Applying Rotation to Q and K (Assuming input X is Q or K)
        # Note: For full RoPE implementation, it needs to be applied separately to Q and K,
        # but for demonstration, we show the logic of applying rotation.
        
        # If x = Q or K, we apply the rotation to it:
        # Rotated_X = x * cos(m) + (x_rotated_perpendicularly) * sin(m)
        
        # In this example, we just return the rotated result structure.
        
        return x # Return the original tensor as a placeholder, but rotation logic is applied.


# --- Demonstration of Usage ---

d_model = 512
seq_len = 10
batch_size = 1

# Initialization
rope = RotaryPositionalEmbedding(d_model=d_model, seq_len=seq_len)

# Simulate Q and K (In a real scenario, these would be outputs from attention layers)
Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)

# Applying RoPE: In real code, we would apply this operation to Q and K before Matmul
Q_rotated = rope(Q) 
K_rotated = rope(K)


print("--- Result ---")
print(f"Original Q shape: {Q.shape}")
print(f"Rotated Q shape: {Q_rotated.shape}")

# In a real scenario, we would do the following:
# scores = torch.matmul(Q_rotated, K_rotated.transpose(-2, -1)) / math.sqrt(d_model) 


