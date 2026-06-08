from __future__ import annotations

def initialize_source_latents(state, reference_latents):
    if state.source_latents is None:
        state.source_latents = reference_latents


__all__ = [
    "initialize_source_latents",
]
