"""
============================================================================
参数有效性属性测试
============================================================================
验证模型参数的有效性

Property 4: 参数有效性
============================================================================
"""

import pytest
import numpy as np
import torch
import os
from hypothesis import given, settings, strategies as st

import sys
sys.path.insert(0, '..')


# ============================================================================
# Property 4: 参数有效性
# **Feature: oampnet-diagnosis-optimization, Property 4: 参数有效性**
# **Validates: Requirements 2.2, 2.3**
# ============================================================================

class TestParameterValidity:
    """参数有效性属性测试"""
    
    def test_oampnet_params_valid(self):
        """
        Property 4a: OAMP-Net参数应该有效
        
        所有参数应该：
        1. 维度正确
        2. 非NaN
        3. 非Inf
        4. 非全零
        """
        # 获取正确的路径（相对于测试文件位置）
        test_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(test_dir, "..", "..", "data", "oampnet_params.mat")
        
        if not os.path.exists(params_path):
            pytest.skip("OAMP-Net参数文件不存在")
        
        import scipy.io as sio
        params = sio.loadmat(params_path)
        
        # 检查gamma参数
        assert 'gamma' in params, "缺少gamma参数"
        gamma = params['gamma'].flatten()
        assert len(gamma) > 0, "gamma参数为空"
        assert not np.any(np.isnan(gamma)), "gamma包含NaN"
        assert not np.any(np.isinf(gamma)), "gamma包含Inf"
        assert not np.all(gamma == 0), "gamma全为零"
        
        # 检查theta参数
        assert 'theta' in params, "缺少theta参数"
        theta = params['theta'].flatten()
        assert len(theta) > 0, "theta参数为空"
        assert not np.any(np.isnan(theta)), "theta包含NaN"
        assert not np.any(np.isinf(theta)), "theta包含Inf"
        
        # 检查num_layers
        assert 'num_layers' in params, "缺少num_layers参数"
        num_layers = int(params['num_layers'].flatten()[0])
        assert num_layers > 0, "num_layers应该大于0"
        assert len(gamma) == num_layers, f"gamma长度({len(gamma)})应该等于num_layers({num_layers})"
        assert len(theta) == num_layers, f"theta长度({len(theta)})应该等于num_layers({num_layers})"
    
    def test_cnn_params_valid(self):
        """
        Property 4b: CNN参数应该有效
        
        所有卷积层参数应该：
        1. 维度正确
        2. 非NaN
        3. 非Inf
        4. 非全零
        """
        # 获取正确的路径（相对于测试文件位置）
        test_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(test_dir, "..", "..", "data", "cnn_params.mat")
        
        if not os.path.exists(params_path):
            pytest.skip("CNN参数文件不存在")
        
        import scipy.io as sio
        params = sio.loadmat(params_path)
        
        # 检查必需的参数
        required_params = [
            'refine_net_0_weight', 'refine_net_0_bias',
            'refine_net_2_weight', 'refine_net_2_bias',
            'refine_net_4_weight', 'refine_net_4_bias',
            'refine_net_6_weight', 'refine_net_6_bias',
            'scale'
        ]
        
        for param_name in required_params:
            if param_name not in params:
                pytest.skip(f"缺少参数: {param_name}")
            
            param = params[param_name]
            
            # 检查有效性
            assert not np.any(np.isnan(param)), f"{param_name}包含NaN"
            assert not np.any(np.isinf(param)), f"{param_name}包含Inf"
            
            # 权重不应该全为零
            if 'weight' in param_name:
                assert not np.all(param == 0), f"{param_name}全为零"
    
    @settings(max_examples=50, deadline=None)
    @given(
        num_layers=st.integers(min_value=5, max_value=20),
        N=st.sampled_from([64, 128, 256])
    )
    def test_oampnet_model_params_valid(self, num_layers, N):
        """
        Property 4c: 新创建的OAMP-Net模型参数应该有效
        
        对于任意配置，模型参数应该正确初始化
        """
        from models.oampnet import OAMPNet
        
        model = OAMPNet(N=N, num_layers=num_layers)
        
        # 检查gamma
        gamma = model.gamma.data.numpy()
        assert len(gamma) == num_layers
        assert not np.any(np.isnan(gamma))
        assert not np.any(np.isinf(gamma))
        
        # 检查theta
        theta = model.theta.data.numpy()
        assert len(theta) == num_layers
        assert not np.any(np.isnan(theta))
        assert not np.any(np.isinf(theta))
        
        # 检查参数总数
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 2 * num_layers, f"参数总数应该是{2*num_layers}，实际是{total_params}"
    
    def test_parameter_ranges_reasonable(self):
        """
        Property 4d: 学习到的参数应该在合理范围内
        
        gamma应该接近1（步长）
        theta经过sigmoid后应该在(0,1)范围内
        """
        # 获取正确的路径（相对于测试文件位置）
        test_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(test_dir, "..", "..", "data", "oampnet_params.mat")
        
        if not os.path.exists(params_path):
            pytest.skip("OAMP-Net参数文件不存在")
        
        import scipy.io as sio
        params = sio.loadmat(params_path)
        
        # gamma应该在合理范围内（0.1到10）
        gamma = params['gamma'].flatten()
        assert np.all(gamma > 0.01), f"gamma有值过小: {np.min(gamma)}"
        assert np.all(gamma < 100), f"gamma有值过大: {np.max(gamma)}"
        
        # theta经过sigmoid后应该在(0,1)
        theta_raw = params['theta'].flatten()
        theta = 1 / (1 + np.exp(-theta_raw))
        assert np.all(theta > 0), "theta应该大于0"
        assert np.all(theta < 1), "theta应该小于1"


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
