import torch as t


class GradScale(t.nn.Module):
    def forward(self, x, scale):
        y = x
        y_grad = x / scale
        return (y - y_grad).detach() + y_grad


class RoundPass(t.nn.Module):
    def forward(self, x):
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad


class Quantize(t.nn.Module):
    def __init__(self, is_activation, bit):
        super(Quantize, self).__init__()
        self.s = t.nn.Parameter(t.zeros(1))
        if is_activation:
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            # signed weight is quantized to [-2^(b-1), 2^(b-1)-1]
            self.thd_neg = - 2 ** (bit - 1)
            self.thd_pos = 2 ** (bit - 1) - 1

        self.grad_scale = GradScale()
        self.round_pass = RoundPass()

    def forward(self, x):
        s_grad_scale = (self.thd_pos * x.numel()) ** 0.5
        s_scale = self.grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = self.round_pass(x)
        x = x * s_scale
        return x


class QuanConv2d(t.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, quan_bit_w=None,
                 quan_bit_a=None, bias=False, **kwargs):
        super(QuanConv2d, self).__init__(in_channels, out_channels, kernel_size, bias=bias, **kwargs)

        if quan_bit_a is None:
            self.quan_a = t.nn.Identity()
        else:
            self.quan_a = Quantize(is_activation=True, bit=quan_bit_a)

        if quan_bit_w is None:
            self.quan_w = t.nn.Identity()
        else:
            self.quan_w = Quantize(is_activation=False, bit=quan_bit_w)

        if bias and (quan_bit_a is not None or quan_bit_w is not None):
            raise Exception('LSQ cannot quantize biases.')

    def forward(self, x):
        weight_quan = self.quan_w(self.weight)
        act_quan = self.quan_a(x)
        return self._conv_forward(act_quan, weight_quan)
