"""ReDimNet2 speaker-embedding model in MLX (inference only).

Port of https://github.com/PalabraAI/redimnet2 (MIT). Parameter names mirror the
PyTorch state_dict keys so conversion is a pure layout transpose (see convert.py).

Layouts used internally (MLX is channels-last):
  2D feature maps: (B, F, T, C)   <->  torch (B, C, F, T)
  1D feature maps: (B, T, F*C)    <->  torch (B, F*C, T)   (same f-major channel order)
"""

import json
import math
from pathlib import Path

import mlx.core as mx
from mlx import nn

# ----------------------------------------------------------------------------
# Primitive layers
# ----------------------------------------------------------------------------


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


class Conv1d(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.weight = mx.zeros((cout, k, cin // groups))
        if bias:
            self.bias = mx.zeros((cout,))
        self._cfg = (stride, padding, groups, bias)

    def __call__(self, x):
        stride, padding, groups, bias = self._cfg
        y = mx.conv1d(x, self.weight, stride=stride, padding=padding, groups=groups)
        return y + self.bias if bias else y


class Conv2d(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        kh, kw = _pair(k)
        self.weight = mx.zeros((cout, kh, kw, cin // groups))
        if bias:
            self.bias = mx.zeros((cout,))
        self._cfg = (_pair(stride), _pair(padding), groups, bias)

    def __call__(self, x):
        stride, padding, groups, bias = self._cfg
        y = mx.conv2d(x, self.weight, stride=stride, padding=padding, groups=groups)
        return y + self.bias if bias else y


class Linear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.weight = mx.zeros((cout, cin))
        self.bias = mx.zeros((cout,))

    def __call__(self, x):
        return mx.addmm(self.bias, x, self.weight.T)


class BatchNorm(nn.Module):
    """Eval-mode BatchNorm over the last (channel) axis."""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((c,))
        self.bias = mx.zeros((c,))
        self.running_mean = mx.zeros((c,))
        self.running_var = mx.ones((c,))
        self._eps = eps

    def __call__(self, x):
        # Same fused form as PyTorch's CPU inference kernel: x * alpha + beta
        alpha = self.weight * mx.rsqrt(self.running_var + self._eps)
        beta = self.bias - self.running_mean * alpha
        return x * alpha + beta


class LayerNormCF(nn.Module):
    """ReDimNet's custom LayerNorm(data_format='channels_first'); channels are last here."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((c,))
        self.bias = mx.zeros((c,))
        self._eps = eps

    def __call__(self, x):
        u = x.mean(-1, keepdims=True)
        s = mx.square(x - u).mean(-1, keepdims=True)
        x = (x - u) / mx.sqrt(s + self._eps)
        return self.weight * x + self.bias


class LayerNorm(nn.Module):
    """torch.nn.LayerNorm over the last axis."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((c,))
        self.bias = mx.zeros((c,))
        self._eps = eps

    def __call__(self, x):
        return mx.fast.layer_norm(x, self.weight, self.bias, self._eps)


class GroupNorm1d(nn.Module):
    """torch.nn.GroupNorm applied to a 1D map, here laid out as (B, T, C)."""

    def __init__(self, groups, c, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((c,))
        self.bias = mx.zeros((c,))
        self._g = groups
        self._eps = eps

    def __call__(self, x):
        B, T, C = x.shape
        g = self._g
        xg = x.reshape(B, T, g, C // g)
        u = xg.mean(axis=(1, 3), keepdims=True)
        v = mx.square(xg - u).mean(axis=(1, 3), keepdims=True)
        xg = (xg - u) * mx.rsqrt(v + self._eps)
        return xg.reshape(B, T, C) * self.weight + self.bias


class Placeholder(nn.Module):
    """Keeps list indices aligned with parameter-free torch Sequential entries."""

    def __init__(self, fn=None):
        super().__init__()
        self._fn = fn

    def __call__(self, x):
        return self._fn(x) if self._fn is not None else x


def gelu_exact(x):
    return x * 0.5 * (1.0 + mx.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3.0))))


def run_seq(layers, x):
    for layer in layers:
        x = layer(x)
    return x


# ----------------------------------------------------------------------------
# ReDimNet structural pieces
# ----------------------------------------------------------------------------


def to1d(x):
    # (B, F, T, C) -> (B, T, F*C)
    B, F, T, C = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, F * C)


def make_to2d(f, c):
    def to2d(x):
        # (B, T, F*C) -> (B, F, T, C)
        B, T, _ = x.shape
        return x.reshape(B, T, f, c).transpose(0, 2, 1, 3)

    return to2d


class Weigth1d(nn.Module):
    def __init__(self, N, C):
        super().__init__()
        self.w = mx.zeros((1, N, C, 1))

    def __call__(self, xs):
        w = mx.softmax(self.w, axis=1)[0, :, :, 0]  # (N, C)
        xs = mx.stack(xs, axis=1)  # (B, N, T, C)
        return (w[None, :, None, :] * xs).sum(axis=1)


def make_upsample(s):
    return (lambda x: mx.repeat(x, s, axis=1)) if s > 1 else (lambda x: x)


class ResBasicBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.conv1pw = Conv2d(c, c, 1)
        self.bn1 = BatchNorm(c)
        self.conv2 = Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.conv2pw = Conv2d(c, c, 1)
        self.bn2 = BatchNorm(c)

    def __call__(self, x):
        out = self.bn1(nn.relu(self.conv1pw(self.conv1(x))))
        out = self.bn2(self.conv2pw(self.conv2(out)))
        return nn.relu(out + x)


class ConvBlock2d(nn.Module):
    def __init__(self, c, block_type):
        super().__init__()
        if block_type != "basic_resnet":
            raise NotImplementedError(f"block_2d_type={block_type!r} not ported yet")
        self.conv_block = ResBasicBlock(c)

    def __call__(self, x):
        return self.conv_block(x)


class ConvNeXtLikeBlock1d(nn.Module):
    """dim=1, norm='bn', norm_placement='mid', activation='gelu', Gdiv=1."""

    def __init__(self, C, kernel_sizes):
        super().__init__()
        self.dwconvs = [Conv1d(C, C, k, padding=k // 2, groups=C) for k in kernel_sizes]
        self.norm = BatchNorm(C * len(kernel_sizes))
        self.pwconv1 = Conv1d(C * len(kernel_sizes), C, 1)

    def __call__(self, x):
        y = mx.concatenate([dw(x) for dw in self.dwconvs], axis=-1)
        y = gelu_exact(self.norm(y))
        return x + self.pwconv1(y)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.k_proj = Linear(dim, dim)
        self.v_proj = Linear(dim, dim)
        self.q_proj = Linear(dim, dim)
        self.out_proj = Linear(dim, dim)
        self._h = heads
        self._scale = (dim // heads) ** -0.5

    def __call__(self, x):
        B, T, D = x.shape
        h = self._h

        def split(t):
            return t.reshape(B, T, h, D // h).transpose(0, 2, 1, 3)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        a = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * self._scale, axis=-1)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(o)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.intermediate_dense = Linear(dim, hidden)
        self.output_dense = Linear(hidden, dim)

    def __call__(self, x):
        return self.output_dense(gelu_new(self.intermediate_dense(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_state, n_mlp, n_head, eps=1e-6):
        super().__init__()
        self.attention = MultiHeadAttention(n_state, n_head)
        self.layer_norm = LayerNorm(n_state, eps)
        self.feed_forward = FeedForward(n_state, n_mlp)
        self.final_layer_norm = LayerNorm(n_state, eps)

    def __call__(self, x):  # x already (B, T, C)
        x = self.layer_norm(x + self.attention(x))
        return self.final_layer_norm(x + self.feed_forward(x))


class TimeContextBlock1d(nn.Module):
    def __init__(self, C, hC, block_type):
        super().__init__()
        if block_type != "conv+att":
            raise NotImplementedError(f"block_1d_type={block_type!r} not ported yet")
        self.red_dim_conv = [Conv1d(C, hC, 1), LayerNormCF(hC)]
        self.tcm = [ConvNeXtLikeBlock1d(hC, [k]) for k in (7, 19, 31, 59)]
        self.tcm.append(TransformerEncoderLayer(hC, hC, 4))
        self.exp_dim_conv = Conv1d(hC, C, 1)

    def __call__(self, x):
        y = run_seq(self.red_dim_conv, x)
        y = run_seq(self.tcm, y)
        return x + self.exp_dim_conv(y)


DEFAULT_STAGES = [
    ((1, 1), 2, 4, [(3, 3)], 24),
    ((2, 1), 3, 3, [(3, 3)], 24),
    ((1, 2), 4, 2, [(3, 3)], 24),
    ((2, 1), 5, 1, [(3, 3)], 24),
    ((1, 2), 4, 1, [(3, 3)], 24),
    ((2, 1), 3, 1, [(3, 3)], 24),
]


class ReDimNet2(nn.Module):
    def __init__(
        self,
        F=72,
        C=24,
        block_1d_type="conv+att",
        block_2d_type="basic_resnet",
        return_2d_output=False,
        out_channels=None,
        stages_setup=None,
        compress_tconvs=True,
        agg_gnorm=False,
        fm_weigthing_type="NC",
        group_divisor=1,
        causal="none",
        dual_agg=False,
        use_freq_pos_enc=False,
        spec_in_channels=1,
        att_dos=None,
    ):
        super().__init__()
        if causal not in ("none", False, None):
            raise NotImplementedError("causal models not ported yet")
        if dual_agg or use_freq_pos_enc or fm_weigthing_type != "NC" or group_divisor != 1:
            raise NotImplementedError("config option not ported yet")
        if block_1d_type.startswith(("v2_", "sdwm_")):
            raise NotImplementedError(f"block_1d_type={block_1d_type!r} not ported yet")
        if att_dos:
            raise NotImplementedError("att_dos not ported yet")
        stages_setup = stages_setup or DEFAULT_STAGES
        agg_groups = C if agg_gnorm is True else (int(agg_gnorm) if agg_gnorm else 0)

        CF = C * F
        self.stem = [Conv2d(spec_in_channels, C, 3, padding=1), LayerNormCF(C), Placeholder(to1d)]
        if agg_groups:
            self.stem_gnorm = GroupNorm1d(agg_groups, CF)

        c, f, stt, max_stt, sft = C, F, 1, 1, 1
        self._num_stages = len(stages_setup)
        for i, (stride, num_blocks, conv_exp, _ks, att_red) in enumerate(stages_setup):
            sf, st = stride
            sft *= sf
            stt *= st
            L = [Weigth1d(i + 1, CF), Placeholder(make_to2d(f, c))]
            cout = int(sf * c * conv_exp)
            groups = math.gcd(c, cout) if compress_tconvs else 1
            L.append(Conv2d(c, cout, (sf, stt), stride=(sf, stt), groups=groups))
            c, f = sf * c, f // sf
            max_stt = max(max_stt, stt)
            for _ in range(num_blocks):
                L.append(ConvBlock2d(c * conv_exp, block_2d_type))
            if conv_exp != 1:
                L.append([Conv2d(c * conv_exp, c, 1), BatchNorm(c, eps=1e-6)])
            L.append(Placeholder(to1d))
            if att_red is not None:
                L.append(TimeContextBlock1d(CF, CF // att_red, block_1d_type))
            L.append(Placeholder(make_upsample(stt)))
            if agg_groups:
                L.append(GroupNorm1d(agg_groups, CF))
            # stage lists mirror the torch nn.Sequential indices (stage{i}.{j}.*)
            setattr(self, f"stage{i}", L)

        self.fin_wght1d = Weigth1d(len(stages_setup) + 1, CF)
        self._time_stride = max_stt
        self._final = (f, c)
        self._return_2d = return_2d_output
        if out_channels is not None:
            self.head = (
                Conv2d(c, out_channels, 1) if return_2d_output else Conv1d(CF, out_channels, 1)
            )
        self._has_head = out_channels is not None

    def _run_stage(self, layers, outs):
        x = layers[0](outs)
        for layer in layers[1:]:
            x = run_seq(layer, x) if isinstance(layer, list) else layer(x)
        return x

    def __call__(self, spec):
        # spec: (B, F, T) log-mel
        T = spec.shape[-1]
        spec = spec[..., : (T // self._time_stride) * self._time_stride]
        x = run_seq(self.stem, spec[..., None])
        if "stem_gnorm" in self:
            x = self.stem_gnorm(x)
        outs = [x]
        for i in range(self._num_stages):
            outs.append(self._run_stage(self[f"stage{i}"], outs))
        x = self.fin_wght1d(outs)
        if self._return_2d:
            f, c = self._final
            x = make_to2d(f, c)(x)  # (B, f, T, c)
        if self._has_head:
            x = self.head(x)
        return x


# ----------------------------------------------------------------------------
# Features (TFMelBanks) and pooling
# ----------------------------------------------------------------------------


class NormalizeAudio(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self._eps = eps

    def __call__(self, x):  # (B, L)
        u = x.mean(-1, keepdims=True)
        sd = mx.sqrt(mx.square(x - u).mean(-1, keepdims=True))
        return (x - u) / (sd + self._eps)


class PreEmphasis(nn.Module):
    def __init__(self):
        super().__init__()
        self.flipped_filter = mx.array([[[-0.97], [1.0]]])  # MLX conv1d layout (1, 2, 1)

    def __call__(self, x):  # (B, L)
        x = mx.concatenate([x[:, 1:2], x], axis=1)  # reflect pad (1, 0)
        return mx.conv1d(x[..., None], self.flipped_filter)[..., 0]


class SpectralFeaturesTF(nn.Module):
    def __init__(self, n_mels, frame_length=400, frame_step=160, n_fft=512, eps=1e-8):
        super().__init__()
        self.real_kernel_pt = mx.zeros((n_fft // 2, frame_length, 1))
        self.image_kernel_pt = mx.zeros((n_fft // 2, frame_length, 1))
        self.melbanks_pt = mx.zeros((n_mels, 1, n_fft // 2))
        self._step = frame_step
        self._eps = eps

    def __call__(self, x):  # (B, L) -> (B, frames, n_mels)
        x = x[..., None]
        pad = self._step // 2
        re = mx.conv1d(x, self.real_kernel_pt, stride=self._step, padding=pad)
        im = mx.conv1d(x, self.image_kernel_pt, stride=self._step, padding=pad)
        p = mx.clip(mx.square(re) + mx.square(im), self._eps, 1 / self._eps)
        mel = mx.conv1d(p, self.melbanks_pt)
        return mx.clip(mel, self._eps, 1 / self._eps)


class TFMelBanks(nn.Module):
    def __init__(self, n_mels, hop_length=160, norm_signal=False, do_preemph=True, eps=1e-8, **_):
        super().__init__()
        self.torchfbank = [
            NormalizeAudio(eps) if norm_signal else Placeholder(),
            PreEmphasis() if do_preemph else Placeholder(),
            SpectralFeaturesTF(n_mels, frame_step=hop_length, eps=eps),
        ]
        self._eps = eps

    def __call__(self, wav):  # (B, L) -> (B, n_mels, frames)
        x = mx.log(run_seq(self.torchfbank, wav) + self._eps)
        x = x - x.mean(axis=1, keepdims=True)
        return x.transpose(0, 2, 1)


class ASTP(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super().__init__()
        self.linear1 = Conv1d(in_dim * 3 if global_context_att else in_dim, bottleneck_dim, 1)
        self.linear2 = Conv1d(bottleneck_dim, in_dim, 1)
        self._gca = global_context_att

    def __call__(self, x):  # (B, T, D) -> (B, 2D)
        if self._gca:
            T = x.shape[1]
            u = x.mean(1, keepdims=True)
            var = mx.square(x - u).sum(1, keepdims=True) / (T - 1)  # torch.var is unbiased
            sd = mx.sqrt(var + 1e-7)
            x_in = mx.concatenate(
                [x, mx.broadcast_to(u, x.shape), mx.broadcast_to(sd, x.shape)], -1
            )
        else:
            x_in = x
        alpha = mx.softmax(self.linear2(mx.tanh(self.linear1(x_in))), axis=1)
        mean = (alpha * x).sum(1)
        var = (alpha * mx.square(x)).sum(1) - mean**2
        std = mx.sqrt(mx.maximum(var, 1e-7))
        return mx.concatenate([mean, std], axis=-1)


class ReDimNet2Wrap(nn.Module):
    def __init__(
        self,
        F=72,
        C=24,
        embed_dim=192,
        hop_length=160,
        pooling_func="ASTP",
        feat_type="tf",
        global_context_att=True,
        emb_bn=False,
        out_channels=None,
        return_2d_output=False,
        spec_params=None,
        pad_right_samples=None,
        before_pool_offset=None,
        num_classes=None,
        feat_agg_dropout=0.0,
        head_activation=None,
        **backbone_kw,
    ):
        super().__init__()
        if feat_type not in ("tf", "tf_mel"):
            raise NotImplementedError(f"feat_type={feat_type!r} not ported yet")
        if pooling_func != "ASTP":
            raise NotImplementedError(f"pooling_func={pooling_func!r} not ported yet")
        self.backbone = ReDimNet2(
            F=F, C=C, out_channels=out_channels, return_2d_output=return_2d_output, **backbone_kw
        )
        sp = dict(spec_params or {})
        sp.pop("do_spec_aug", None)
        self.spec = TFMelBanks(n_mels=F, hop_length=hop_length, **sp)
        if out_channels is None:
            pool_in = C * F
        elif return_2d_output:
            pool_in = self.backbone._final[0] * out_channels
        else:
            pool_in = out_channels
        self.pool = ASTP(pool_in, global_context_att=global_context_att)
        self.bn = BatchNorm(2 * pool_in)
        self.linear = Linear(2 * pool_in, embed_dim)
        if emb_bn:
            self.bn2 = BatchNorm(embed_dim)
        self._pad_right = pad_right_samples
        self._offset = before_pool_offset

    def __call__(self, wav):
        """wav: (B, L) float32 16 kHz -> (B, embed_dim) embedding."""
        if self._pad_right:
            wav = mx.pad(wav, [(0, 0), (0, self._pad_right)])
        out = self.backbone(self.spec(wav))
        if out.ndim == 4:
            # torch reshapes (B, C, F, T) -> (B, C*F, T): channel-major order
            B, f, T, c = out.shape
            out = out.transpose(0, 2, 3, 1).reshape(B, T, c * f)
        if self._offset is not None:
            out = out[:, self._offset :]
        out = self.linear(self.bn(self.pool(out)))
        if "bn2" in self:
            out = self.bn2(out)
        return out


def resolve_model_path(path_or_repo, revision=None):
    """Return a local directory for a converted model.

    Accepts a local directory, or a Hugging Face repo id (e.g.
    "causal/redimnet2-b3-vox2-lm-mlx") which is downloaded into the HF cache.
    """
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(path_or_repo),
            revision=revision,
            allow_patterns=["config.json", "weights.safetensors"],
        )
    )


def load_model(path_or_repo, revision=None):
    """Load a converted model from a local directory or a Hugging Face repo id."""
    path = resolve_model_path(path_or_repo, revision)
    cfg = json.loads((path / "config.json").read_text())
    model = ReDimNet2Wrap(**cfg)
    model.load_weights(str(path / "weights.safetensors"), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
