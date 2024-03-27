import torch
from parameterized import parameterized

from dolomite_engine.hf_models.modeling_utils import get_activation_function
from dolomite_engine.hf_models.modeling_utils.activations import get_base_activation
from dolomite_engine.hf_models.modeling_utils.activations.glu import GLUActivation

from ..test_common import TestCommons


class ActivationsTest(TestCommons):
    @parameterized.expand(TestCommons.make_args_matrix(TestCommons.get_all_devices()))
    def test_math_gelu_equivalence(self, device: torch.device) -> None:
        self.skip_test_if_device_unavailable(device)

        math_gelu = get_activation_function("gelu_math_tanh")
        pytorch_gelu = get_activation_function("gelu_pytorch_tanh")

        math_gelu_input = torch.randn(10, 10, device=device, requires_grad=True)
        pytorch_gelu_input = torch.empty_like(math_gelu_input, requires_grad=True)
        pytorch_gelu_input.data = math_gelu_input.data

        math_gelu_output = math_gelu(math_gelu_input)
        pytorch_gelu_output = pytorch_gelu(pytorch_gelu_input)

        math_gelu_output.sum().backward()
        pytorch_gelu_output.sum().backward()

        self.assert_equal_tensors(math_gelu_output, pytorch_gelu_output, False)
        self.assert_equal_tensors(math_gelu_input.grad, pytorch_gelu_input.grad, False)

    @parameterized.expand(TestCommons.make_args_matrix(TestCommons.get_all_devices()))
    def test_sigmoid_glu(self, device: torch.device) -> None:
        self.skip_test_if_device_unavailable(device)

        base_sigmoid = get_base_activation("sigmoid")
        sigmoid_glu = GLUActivation(base_sigmoid)

        pytorch_glu = get_activation_function("glu")

        x = torch.randn(10, 10, device=device)
        sigmoid_glu_output = sigmoid_glu(x)
        pytorch_glu_output = pytorch_glu(x)

        self.assert_equal_tensors(sigmoid_glu_output, pytorch_glu_output, True)
