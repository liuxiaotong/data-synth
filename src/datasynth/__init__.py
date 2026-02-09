"""DataSynth - 数据合成工具

基于种子数据和 Schema 批量生成高质量训练数据。
"""

__version__ = "0.4.0"

from datasynth.synthesizer import DataSynthesizer, SynthesisResult
from datasynth.config import SynthesisConfig

__all__ = ["DataSynthesizer", "SynthesisResult", "SynthesisConfig", "__version__"]
