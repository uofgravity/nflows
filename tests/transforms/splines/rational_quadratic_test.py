import torch
import torchtestcase

from nflows.transforms import splines


class RationalQuadraticSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins)
        unnormalized_derivatives = torch.randn(*shape, num_bins + 1)

        def call_spline_fn(inputs, inverse=False):
            return splines.rational_quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=inverse,
            )

        inputs = torch.rand(*shape)
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = 1e-4
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))


class UnconstrainedRationalQuadraticSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins)
        unnormalized_derivatives = torch.randn(*shape, num_bins + 1)

        def call_spline_fn(inputs, inverse=False):
            return splines.unconstrained_rational_quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=inverse,
            )

        inputs = 3 * torch.randn(*shape)  # Note inputs are outside [0,1].
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = 1e-4
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_forward_inverse_are_consistent_in_tails(self):
        num_bins = 10
        shape = [2, 3, 4]
        tail_bound = 1.0

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins)
        unnormalized_derivatives = torch.randn(*shape, num_bins + 1)

        def call_spline_fn(inputs, inverse=False):
            return splines.unconstrained_rational_quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=inverse,
            )

        inputs = torch.sign(torch.randn(*shape)) * (tail_bound + torch.rand(*shape))  # Now *all* inputs are outside [-tail_bound, tail_bound].
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = 1e-4
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))


def _reference_unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tail_bound=1.0,
):
    """The previous (masked-indexing) implementation, kept as a numerical reference."""
    import numpy as np
    from torch.nn import functional as F

    from nflows.transforms.splines.rational_quadratic import (
        DEFAULT_MIN_DERIVATIVE,
        rational_quadratic_spline,
    )

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0
    if torch.any(inside_interval_mask):
        outputs_masked, logabsdet_masked = rational_quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
        )
        outputs[inside_interval_mask] = outputs_masked
        logabsdet[inside_interval_mask] = logabsdet_masked
    return outputs, logabsdet


class UnconstrainedRationalQuadraticSplineStaticShapeTest(torchtestcase.TorchTestCase):
    """The unconstrained spline is written with static shapes (clamp + where) so it
    can be captured by torch.compile. It must remain bit-identical to the masked
    reference implementation, including gradients."""

    def _make_inputs(self, dtype, mode):
        torch.manual_seed(1234)
        num_bins = 8
        shape = [16, 5]
        if mode == "mixed":
            inputs = 3 * torch.randn(*shape, dtype=dtype)
        elif mode == "all_outside":
            inputs = torch.sign(torch.randn(*shape)) * (1.0 + torch.rand(*shape))
            inputs = inputs.to(dtype)
        elif mode == "all_inside":
            inputs = torch.rand(*shape, dtype=dtype) * 2 - 1
        elif mode == "boundary":
            inputs = torch.tensor([-1.0, 1.0, 0.0, -1.0, 1.0] * 16, dtype=dtype).reshape(shape)
        params = [
            torch.randn(*shape, num_bins, dtype=dtype),
            torch.randn(*shape, num_bins, dtype=dtype),
            torch.randn(*shape, num_bins - 1, dtype=dtype),
        ]
        return inputs, params

    def _check(self, dtype, mode, inverse):
        inputs, params = self._make_inputs(dtype, mode)

        def run(fn):
            x = inputs.clone().requires_grad_(True)
            ps = [p.clone().requires_grad_(True) for p in params]
            out, logabsdet = fn(x, *ps, inverse=inverse)
            (out.sum() + logabsdet.sum()).backward()
            return out, logabsdet, [x.grad] + [p.grad for p in ps]

        ref = run(_reference_unconstrained_rational_quadratic_spline)
        new = run(splines.unconstrained_rational_quadratic_spline)
        self.assertEqual(ref[0].dtype, new[0].dtype)
        self.assertTrue(torch.equal(ref[0], new[0]), "outputs differ")
        self.assertTrue(torch.equal(ref[1], new[1]), "logabsdet differ")
        for g_ref, g_new in zip(ref[2], new[2]):
            # When every input lies in the tails the reference never touches the
            # spline parameters (grad None); the static-shape version returns an
            # explicit zero gradient, which is equivalent.
            if g_ref is None:
                self.assertTrue(g_new is None or not g_new.any(), "gradients differ")
            else:
                self.assertTrue(torch.equal(g_ref, g_new), "gradients differ")

    def test_matches_reference_implementation(self):
        for dtype in (torch.float32, torch.float64):
            for mode in ("mixed", "all_outside", "all_inside", "boundary"):
                for inverse in (False, True):
                    with self.subTest(dtype=dtype, mode=mode, inverse=inverse):
                        self._check(dtype, mode, inverse)

    def test_compiles_without_graph_breaks(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile not available")
        inputs, params = self._make_inputs(torch.float32, "mixed")
        for inverse in (False, True):
            eager_out, eager_logabsdet = splines.unconstrained_rational_quadratic_spline(
                inputs, *params, inverse=inverse
            )
            compiled = torch.compile(
                splines.unconstrained_rational_quadratic_spline,
                fullgraph=True,
                backend="aot_eager",
            )
            out, logabsdet = compiled(inputs, *params, inverse=inverse)
            self.eps = 1e-6
            self.assertEqual(out, eager_out)
            self.assertEqual(logabsdet, eager_logabsdet)

    def test_inner_spline_domain_check_is_optional(self):
        from nflows.transforms.base import InputOutsideDomain

        num_bins = 8
        shape = [4, 3]
        params = [
            torch.randn(*shape, num_bins),
            torch.randn(*shape, num_bins),
            torch.randn(*shape, num_bins + 1),
        ]
        inputs = torch.rand(*shape) + 1.5  # outside [0, 1]
        with self.assertRaises(InputOutsideDomain):
            splines.rational_quadratic_spline(inputs, *params)
        with self.assertRaises(InputOutsideDomain):
            splines.rational_quadratic_spline(inputs, *params, check_domain=True)
        # The caller takes responsibility for the domain; no host-side check.
        splines.rational_quadratic_spline(inputs.clamp(0, 1), *params, check_domain=False)
