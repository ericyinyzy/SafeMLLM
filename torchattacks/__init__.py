"""
Slimmed-down version of torchattacks (https://github.com/Harry24k/adversarial-attacks-pytorch)
for SafeMLLM. Only the PGD UAP variant adapted for LLaVA-1.5 is kept.
"""
from .attack import Attack
from .attacks.pgd_uap_llava import PGD

__version__ = "3.5.1+safemllm"
__all__ = ["Attack", "PGD"]
