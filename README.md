# arxiv.2505.09388
Qwen3 MoE Model Simplified Implementation following the technical report

# Implemented Components:

Core Architecture:
Decoder-only transformer based on Llama architecture
RMSNorm for all normalization layers
SwiGLU activation function in feed-forward networks
Rotary Positional Embeddings (RoPE) for position encoding
Grouped Query Attention (GQA) for efficient attention

MoE Specifics:
Configurable number of experts (e.g., 64)
Top-k expert selection (e.g., 12 experts per token)
Shared expert that's always activated alongside sparse experts
MoE layers replace FFN layers at configurable frequency
Gating mechanism for expert selection

Key Features:
Flexible configuration via QwenMoEConfig
Causal language modeling head
KV caching for efficient generation
Proper attention masking
Model parallelism ready

# Questions for Clarification:

Expert Selection Strategy: Should I implement any specific load balancing mechanism for the experts (like auxiliary losses to encourage balanced expert usage)?

Shared Expert Integration: How should the shared expert output be combined with the sparse expert outputs? Currently, I'm using simple addition - is this correct?

MoE Gating: Should the gating network use any specific initialization or training techniques?

Expert Capacity: Do you need expert capacity limiting (dropping tokens when experts are overloaded)?

Model Sizes: What are the specific parameter counts for different model variants (e.g., 7B, 14B, 72B)?

Training Specifics: Any specific training techniques like expert dropout or noise injection during gating?
