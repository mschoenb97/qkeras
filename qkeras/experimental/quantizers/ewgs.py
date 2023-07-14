"""
Element-wise gradient scaling quantizer.

See https://arxiv.org/pdf/2104.00903.pdf
"""

import sys
import os

# update path so that we have access to qkeras
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from qkeras import quantized_linear

print(quantized_linear)