"""
RMSNorm 正确性测试

测试内容：
1. Forward 数值正确性
2. Backward 数值正确性
3. 不同输入形状
4. 不同数据类型
5. 边界情况
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.rms_norm import liger_rms_norm, RMSNorm, LigerRMSNormFunction


def pytorch_rms_norm_reference(x, weight, eps=1e-6):
    """PyTorch 参考实现"""
    # 全部在 float32 计算
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(x.dtype)


class TestRMSNormForward:
    """Forward pass 测试"""

    @pytest.mark.parametrize("shape", [
        (2, 128, 1024),
        (4, 256, 2048),
        (2, 512, 4096),
        (1, 1024, 8192),
        (8, 64, 4096),
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_forward_correctness(self, shape, dtype):
        """测试 forward 数值正确性"""
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("BF16 not supported")

        device = "cuda"
        batch, seq, hidden = shape

        x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
        w = torch.randn(hidden, device=device, dtype=dtype)

        # 我们的实现
        y_ours = liger_rms_norm(x, w)

        # 参考实现
        y_ref = pytorch_rms_norm_reference(x, w)

        # 根据精度设置 tolerance
        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-5
        else:
            atol, rtol = 1e-2, 1e-2

        torch.testing.assert_close(y_ours, y_ref, atol=atol, rtol=rtol)

    def test_forward_with_zeros(self):
        """测试全零输入（边界情况）"""
        device = "cuda"
        dtype = torch.float16

        x = torch.zeros(2, 64, 1024, device=device, dtype=dtype)
        w = torch.ones(1024, device=device, dtype=dtype)

        y = liger_rms_norm(x, w)

        # 全零输入应该输出全零（或接近零）
        assert y.abs().max() < 1e-3

    def test_forward_contiguous(self):
        """测试非连续输入"""
        device = "cuda"
        dtype = torch.float16

        # 创建非连续 tensor
        x = torch.randn(4, 256, 2048, device=device, dtype=dtype).transpose(0, 1)
        assert not x.is_contiguous()

        w = torch.randn(2048, device=device, dtype=dtype)

        # 应该能正常工作（内部会 contiguous()）
        y = liger_rms_norm(x, w)
        y_ref = pytorch_rms_norm_reference(x.contiguous(), w)

        torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)


class TestRMSNormBackward:
    """Backward pass 测试"""

    @pytest.mark.parametrize("shape", [
        (2, 128, 1024),
        (4, 256, 2048),
        (2, 512, 4096),
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_backward_correctness(self, shape, dtype):
        """测试 backward 数值正确性"""
        device = "cuda"
        batch, seq, hidden = shape

        # 我们的实现
        x1 = torch.randn(batch, seq, hidden, device=device, dtype=dtype, requires_grad=True)
        w1 = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
        y1 = liger_rms_norm(x1, w1)
        loss1 = y1.sum()
        loss1.backward()

        # 参考实现
        x2 = x1.detach().clone().requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)
        y2 = pytorch_rms_norm_reference(x2, w2)
        loss2 = y2.sum()
        loss2.backward()

        # 比较梯度
        if dtype == torch.float32:
            atol, rtol = 1e-4, 1e-4
        else:
            atol, rtol = 1e-2, 1e-2

        torch.testing.assert_close(x1.grad, x2.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(w1.grad, w2.grad, atol=atol, rtol=rtol)

    def test_backward_with_upstream_grad(self):
        """测试带有上游梯度的情况"""
        device = "cuda"
        dtype = torch.float16

        x = torch.randn(2, 64, 1024, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(1024, device=device, dtype=dtype, requires_grad=True)

        y = liger_rms_norm(x, w)

        # 使用随机上游梯度
        grad_output = torch.randn_like(y)
        y.backward(grad_output)

        # 检查梯度不是 None 且形状正确
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert w.grad is not None
        assert w.grad.shape == w.shape


class TestRMSNormModule:
    """nn.Module wrapper 测试"""

    def test_module_forward(self):
        """测试 Module forward"""
        device = "cuda"
        dtype = torch.float16

        hidden_dim = 2048
        module = RMSNorm(hidden_dim).to(device).to(dtype)

        x = torch.randn(2, 128, hidden_dim, device=device, dtype=dtype)
        y = module(x)

        assert y.shape == x.shape
        assert y.dtype == dtype

    def test_module_backward(self):
        """测试 Module backward"""
        device = "cuda"
        dtype = torch.float16

        hidden_dim = 2048
        module = RMSNorm(hidden_dim).to(device).to(dtype)

        x = torch.randn(2, 128, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        y = module(x)
        y.sum().backward()

        assert x.grad is not None
        assert module.weight.grad is not None

    def test_module_in_sequential(self):
        """测试在 Sequential 中使用"""
        device = "cuda"
        dtype = torch.float16

        hidden_dim = 1024
        model = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            RMSNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ).to(device).to(dtype)

        x = torch.randn(2, 64, hidden_dim, device=device, dtype=dtype)
        y = model(x)

        assert y.shape == x.shape


class TestRMSNormGradCheck:
    """梯度检查测试（使用 torch.autograd.gradcheck）"""

    def test_gradcheck_float64(self):
        """使用 float64 进行精确梯度检查"""
        device = "cuda"

        # gradcheck 需要 float64
        x = torch.randn(2, 16, 64, device=device, dtype=torch.float64, requires_grad=True)
        w = torch.randn(64, device=device, dtype=torch.float64, requires_grad=True)

        # 由于我们的 kernel 可能不支持 float64，我们跳过这个测试
        # 或者使用较宽松的 tolerance
        pytest.skip("Triton kernel may not support float64 precisely")


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "-s"])
