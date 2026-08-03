"""
RUS — Remove Ur Refusal
Living Ablation Engine v1.2.0

Quickstart:
    import rus
    rus.ablate("Qwen/Qwen2.5-7B-Instruct", load_in_8bit=True)

Step-by-step:
    from rus import RusEngine
    engine = RusEngine("Qwen/Qwen2.5-3B-Instruct", load_in_8bit=True)
    engine.analyze()
    engine.show_refusal()
    engine.ablate(k=5)
    engine.compare()
    engine.save()

CLI:
    python -m rus Qwen/Qwen2.5-3B-Instruct --8bit -n 48 -k 5 -c 0.8
"""

__version__ = "1.2.0"

from .api import RusEngine, ablate
from .cli import main  # for console script entry point
