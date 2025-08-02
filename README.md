# arxiv.2505.09388
Qwen3 MoE Model Simplified Implementation following the technical report

# Implemented Components:

Core Architecture:
1. Decoder-only transformer based on Llama architecture
2. RMSNorm for all normalization layers
3. SwiGLU activation function in feed-forward networks
4. Rotary Positional Embeddings (RoPE) for position encoding
5. Grouped Query Attention (GQA) for efficient attention

MoE Specifics:
1. Configurable number of experts (e.g., 64)
2. Top-k expert selection (e.g., 12 experts per token)
3. Shared expert that's always activated alongside sparse experts
4. MoE layers replace FFN layers at configurable frequency
5. Gating mechanism for expert selection

Key Features:
1. Flexible configuration via QwenMoEConfig
2. Causal language modeling head
3. KV caching for efficient generation
4. Proper attention masking
5. Model parallelism ready

# Questions for Clarification:

Expert Selection Strategy: Should I implement any specific load balancing mechanism for the experts (like auxiliary losses to encourage balanced expert usage)?

Shared Expert Integration: How should the shared expert output be combined with the sparse expert outputs? Currently, I'm using simple addition - is this correct?

MoE Gating: Should the gating network use any specific initialization or training techniques?

Expert Capacity: Do you need expert capacity limiting (dropping tokens when experts are overloaded)?

Model Sizes: What are the specific parameter counts for different model variants (e.g., 7B, 14B, 72B)?

Training Specifics: Any specific training techniques like expert dropout or noise injection during gating?
