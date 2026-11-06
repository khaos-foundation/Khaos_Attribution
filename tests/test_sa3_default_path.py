"""The served SA3 measurement is the worker's default path: ``_gradient`` (vnorm and loss targets), ``_loss``,
``_squared`` at pool 1 and ``_forward`` on a toy model, at the default arguments, against values frozen from the
code before the research options (latent pooling, noise draws, shared noise, block sketches) were added."""
import numpy as np
import pytest

from khaos_attribution.engines.sa3 import measure_worker as W

TIMESTEPS = [0.1, 0.5, 0.9]
CHANNELS, FRAMES, VALID = 3, 11, 8


class _NoSwitch:
    def set_lora_strength(self, s):
        raise AssertionError("the default targets never switch the adapter")


def _toy(torch):
    """A 3-channel toy with 12 parameters and numpy-seeded inputs: (model, params, clip, wrapped)."""
    rng = np.random.default_rng(0)

    def draw(*shape):
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(draw(CHANNELS, CHANNELS))
            self.v = torch.nn.Parameter(draw(CHANNELS))

        def forward(self, x, t, cond=None, cfg_dropout_prob=0.0, padding_mask=None, **kw):
            assert kw == {}, "the default forward passes nothing beyond the trainer's arguments"
            h = torch.einsum("bct,cd->bdt", x, self.w)
            return torch.tanh(h + cond["emb"] + self.v.view(1, -1, 1) * t.view(1, 1, 1))

    model = Toy()
    latents, noise = draw(1, CHANNELS, FRAMES), draw(1, CHANNELS, FRAMES)
    cond = {"emb": draw(1, CHANNELS, 1)}
    mask = torch.zeros(1, FRAMES, dtype=torch.bool)
    mask[:, :VALID] = True
    return model, [model.w, model.v], (latents, cond, mask, noise), _NoSwitch()


# Produced by 60db9c3's measure_worker.py on this toy (CPU, float32); the pin for the served arithmetic.
GRADIENTS = {
    ('vnorm', False): [
        -0.026705002412199974, 0.0011877575889229774, 0.04951750114560127, -0.0037281259428709745,
        0.016807829961180687, 0.08185064792633057, 0.09737320989370346, -0.023280000314116478, -0.13217902183532715,
        0.01663694530725479, 0.023381879553198814, 0.005549496039748192
    ],
    ('vnorm', True): [
        -0.13333849608898163, 0.00488702580332756, 0.24407367408275604, -0.016944877803325653, 0.08327173441648483,
        0.3982473611831665, 0.4694399833679199, -0.1034536138176918, -0.635082483291626, 0.078216053545475,
        0.1107167974114418, 0.029256602749228477
    ],
    ('loss', False): [
        0.0018401741981506348, 0.11119765788316727, 0.060526471585035324, 0.05232984200119972, -0.14678572118282318,
        0.011686244048178196, -0.15976063907146454, -0.03678019344806671, -0.19924049079418182, -0.262369841337204,
        0.009057003073394299, -0.023012449964880943
    ],
    ('loss', True): [
        0.08257655054330826, 0.1457810252904892, 0.0938597023487091, 0.08821222931146622, -0.22214503586292267,
        0.026229092851281166, -0.2510654926300049, -0.0585508830845356, -0.3471943140029907, -0.4025912284851074,
        0.018410583958029747, -0.03478188440203667
    ],
}
LOSS = 2.49040158589681
SQUARED = 1.695070743560791
FORWARD = [
    0.20194943249225616, 0.35010668635368347, -0.9522589445114136, -0.46246325969696045, 0.517402172088623,
    0.871816098690033, -0.9302190542221069, 0.9431639313697815, 0.3712158799171448, 0.17387375235557556,
    0.31178387999534607, 0.9823070764541626, 0.9405481219291687, 0.7390770316123962, 0.9719117283821106,
    0.994463324546814, 0.9980476498603821, 0.42426374554634094, 0.9987753033638, 0.9817555546760559,
    0.9743327498435974, 0.9512719511985779, -0.8179505467414856, 0.36129266023635864, 0.6054044961929321,
    0.011679131537675858, -0.538180947303772, -0.7866630554199219, 0.8172249794006348, -0.648309051990509,
    -0.1313031017780304, 0.24716657400131226, -0.04036479443311691
]


def _close(got, want):
    # Not bit-exact: a 3x3 matmul may reduce in another order on another CPU; a change of rule is far larger.
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64),
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("target,timestep_norm", list(GRADIENTS))
def test_the_default_gradient_is_the_frozen_one(target, timestep_norm):
    torch = pytest.importorskip("torch")
    model, params, clip, wrapped = _toy(torch)
    flat, residual = W._gradient(model, torch, params, clip, TIMESTEPS, target, wrapped=wrapped,
                                 timestep_norm=timestep_norm)
    assert flat.dtype == np.float32 and flat.shape == (CHANNELS * CHANNELS + CHANNELS,)
    _close(flat, GRADIENTS[(target, timestep_norm)])
    assert np.isfinite(residual)


def test_the_loss_the_square_and_the_forward_are_the_frozen_ones():
    torch = pytest.importorskip("torch")
    model, _params, clip, _wrapped = _toy(torch)
    latents, cond, mask, noise = clip
    _close([W._loss(model, torch, clip, TIMESTEPS)], [LOSS])
    _close([float(W._squared(torch, latents - noise, mask.unsqueeze(1)))], [SQUARED])
    out = W._forward(model, torch, latents, cond, mask, noise, 0.3)
    assert out.dtype == torch.float32 and tuple(out.shape) == (1, CHANNELS, FRAMES)
    _close(out.reshape(-1).tolist(), FORWARD)
