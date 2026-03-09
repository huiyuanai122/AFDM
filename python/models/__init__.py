# Detection Models
from .oampnet import OAMPNet, OAMPNetV2, soft_qpsk_projection
from .cnn_detector import CNNDetector

# 向后兼容
HybridOAMPNet = OAMPNet
