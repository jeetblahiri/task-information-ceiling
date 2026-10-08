#!/usr/bin/env python3
"""Independent algebra/likelihood audit; imports no physical study/core code."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import null_space, solve_triangular
from scipy.special import expit, logsumexp, xlogy
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026100841
rng = np.random.default_rng(SEED)


def h2(p):
    return -(xlogy(p, p) + xlogy(1-p, 1-p)) / np.log(2)


def gaussian_information(d, nodes=160):
    z, w = hermgauss(nodes)
    p = expit(d * (d/2 + np.sqrt(2)*z))
    return float(1 - w @ h2(p) / np.sqrt(np.pi))


def posterior(x, plus, minus, covariance, weights):
    chol = np.linalg.cholesky(covariance)
    wx = solve_triangular(chol, np.asarray(x).T, lower=True).T
    wp = solve_triangular(chol, np.asarray(plus).T, lower=True).T
    wm = solve_triangular(chol, np.asarray(minus).T, lower=True).T
    lw = np.log(weights)
    # The common x quadratic and common determinant cancel from class odds.
    lp = logsumexp(wx @ wp.T - .5*np.sum(wp*wp, axis=1) + lw, axis=1)
    lm = logsumexp(wx @ wm.T - .5*np.sum(wm*wm, axis=1) + lw, axis=1)
    return expit(lp-lm)


def summary(values, width):
    v = np.asarray(values)
    return {"mean": float(v.mean()), "SE": float(v.std(ddof=1)/np.sqrt(len(v))),
            "pointwise_Hoeffding95_halfwidth": float(width*np.sqrt(np.log(40)/(2*len(v)))),
            "independent_outer_draws": len(v)}


def car_and_averaging():
    m, t = 7, 12
    qs = null_space(np.ones((1, m)))
    qo = null_space(np.ones((1, 3)))
    ss = .4 ** np.abs(np.arange(m)[:, None]-np.arange(m)[None, :])
    st = .65 ** np.abs(np.arange(t)[:, None]-np.arange(t)[None, :])
    groups = [[0, 1], [2, 3], [4, 5, 6]]
    a = np.zeros((3, m))
    for j, group in enumerate(groups):
        a[j, group] = 1/len(group)
    b = np.zeros((4, t))
    for j in range(4):
        b[j, 3*j:3*j+3] = 1/3
    car = np.kron(np.eye(t), qs.T)
    raw_map = np.kron(b, qo.T @ a)
    map_after_car = np.kron(b, qo.T @ a @ qs)
    raw_cov = np.kron(st, ss)
    car_cov = car @ raw_cov @ car.T
    cov_direct = raw_map @ raw_cov @ raw_map.T
    cov_car = map_after_car @ car_cov @ map_after_car.T
    cov_separable = np.kron(b @ st @ b.T, qo.T @ a @ ss @ a.T @ qo)
    sample = rng.normal(size=(m, t))
    direct = qo.T @ a @ sample @ b.T
    vector_error = np.max(np.abs(direct.ravel(order="F") - raw_map @ sample.ravel(order="F")))
    car_error = np.max(np.abs(raw_map - map_after_car @ car))
    covariance_error = max(np.max(np.abs(cov_direct-cov_car)), np.max(np.abs(cov_direct-cov_separable)))
    contrasts = rng.normal(size=(80, car.shape[0]))
    d_full = np.einsum("ni,ij,nj->n", contrasts, np.linalg.inv(car_cov), contrasts)
    projected = contrasts @ map_after_car.T
    d_projected = np.einsum("ni,ij,nj->n", projected, np.linalg.inv(cov_car), projected)
    assert vector_error < 1e-13 and car_error < 1e-13 and covariance_error < 1e-13
    assert np.all(d_projected <= d_full + 1e-10)
    return {"vectorization": "column-major vec_F: time blocks of sensor coordinates",
            "raw_sensor_time_dimensions": [m, t], "full_CAR_rank": int(np.linalg.matrix_rank(car)),
            "compressed_output_rank": int(np.linalg.matrix_rank(map_after_car)),
            "raw_map_from_CAR_max_error": float(car_error), "vec_Kronecker_max_error": float(vector_error),
            "covariance_propagation_max_error": float(covariance_error),
            "minimum_projected_covariance_eigenvalue": float(np.linalg.eigvalsh(cov_car).min()),
            "three_sample_AR065_average_variance": float((b @ st @ b.T)[0, 0]),
            "wrong_iid_variance": 1/3,
            "max_neighbor_average_covariance": float(np.diag(b @ st @ b.T, 1).max()),
            "max_projected_minus_full_squared_separation": float((d_projected-d_full).max())}


def midpoint_mixture():
    # A hidden sign state changes both the contrast direction and a class-shared
    # midpoint cue. Contrasts alone span x1; all affine means span both axes.
    a, b = 2., 2.
    plus2 = np.array([[a/2, b], [-a/2, -b]])
    minus2 = np.array([[-a/2, b], [a/2, -b]])
    weights = np.array([.5, .5])
    q = 7
    orientation, _ = np.linalg.qr(rng.normal(size=(q, 2)))
    temp = rng.normal(size=(q, q))
    cov = temp @ temp.T/q + np.eye(q)
    chol = np.linalg.cholesky(cov)
    common = rng.normal(size=q)
    plus = common + plus2 @ orientation.T @ chol.T
    minus = common + minus2 @ orientation.T @ chol.T
    ref = plus[0]
    all_white = solve_triangular(chol, (np.vstack([plus, minus])-ref).T, lower=True).T
    _, singular, vt = np.linalg.svd(all_white, full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-12))
    reduce = vt[:rank] @ np.linalg.inv(chol)
    contrast_reduce = orientation[:, :1].T @ np.linalg.inv(chol)
    n = 65536
    heads = rng.integers(0, 2, size=n)
    labels = rng.integers(0, 2, size=n)*2-1
    means = np.where(labels[:, None] > 0, plus[heads], minus[heads])
    samples = means + rng.normal(size=(n, q)) @ chol.T
    full_p = posterior(samples, plus, minus, cov, weights)
    reduced_p = posterior(samples @ reduce.T, plus @ reduce.T, minus @ reduce.T, reduce @ cov @ reduce.T, weights)
    contrast_p = posterior(samples @ contrast_reduce.T, plus @ contrast_reduce.T, minus @ contrast_reduce.T,
                           contrast_reduce @ cov @ contrast_reduce.T, weights)
    info, risk = 1-h2(full_p), np.minimum(full_p, 1-full_p)
    projected_info = 1-h2(contrast_p)
    rotation, _ = np.linalg.qr(rng.normal(size=(q, q)))
    inverse_p = posterior(samples @ rotation.T, plus @ rotation.T, minus @ rotation.T, rotation @ cov @ rotation.T, weights)
    posterior_error = float(np.max(np.abs(full_p-reduced_p)))
    rotation_error = float(np.max(np.abs(full_p-inverse_p)))
    assert rank == 2 and posterior_error < 1e-12 and rotation_error < 1e-12
    assert np.max(np.abs(contrast_p-.5)) < 1e-12
    quadrature = []
    for nodes in [48, 96, 160]:
        z, w = hermgauss(nodes)
        noise = np.stack(np.meshgrid(np.sqrt(2)*z, np.sqrt(2)*z, indexing="ij"), axis=-1).reshape(-1, 2)
        grid_weight = np.outer(w, w).ravel()/np.pi
        mi = 0.
        for sign_means in [plus2, minus2]:
            for h in range(2):
                p = posterior(noise+sign_means[h], plus2, minus2, np.eye(2), weights)
                mi += .25*float(grid_weight @ (1-h2(p)))
        quadrature.append({"nodes_per_axis": nodes, "MI_bits": mi})
    p1, p2 = norm.cdf(-a/2), norm.cdf(-b)
    exact_risk = float(p1+p2-2*p1*p2)
    assert abs(risk.mean()-exact_risk) < summary(risk, .5)["pointwise_Hoeffding95_halfwidth"]
    assert abs(info.mean()-quadrature[-1]["MI_bits"]) < summary(info, 1)["pointwise_Hoeffding95_halfwidth"]
    return {"all_affine_mean_span_rank": rank, "within_head_contrast_span_rank": 1,
            "sufficient_reduction_posterior_max_error": posterior_error,
            "invertible_coordinate_change_posterior_max_error": rotation_error,
            "contrast_only_projection_MI_exact": 0., "hidden_MI_MC": summary(info, 1),
            "hidden_MI_quadrature": quadrature, "known_head_MI_bits": gaussian_information(a),
            "known_head_error": float(p1), "hidden_risk_exact": exact_risk,
            "hidden_risk_MC": summary(risk, .5),
            "paired_information_loss": summary(info-projected_info, 2),
            "paired_risk_increase": summary(.5-risk, 1)}


def predictions_and_decoder():
    development = np.array([2., .5])
    heldout = np.array([.5, 2.])
    prediction = [gaussian_information(d) for d in development]
    actual = [gaussian_information(d) for d in heldout]
    # A fitted probability map can have the correct decision boundary yet
    # assign poor confidence. Its log-loss score is a lower bound, not MI.
    d, fitted_slope = .5, 4.
    z, w = hermgauss(180)
    xplus = d/2 + np.sqrt(2)*z
    ce = float(w @ (np.logaddexp(0, -fitted_slope*xplus)/np.log(2))/np.sqrt(np.pi))
    actual_mi = gaussian_information(d)
    error = float(norm.cdf(-d/2))
    fano_lower = float(1-h2(error))
    assert np.argmax(prediction) != np.argmax(actual)
    assert 1-ce <= actual_mi + 1e-12 and fano_lower <= actual_mi + 1e-12
    return {"development_prediction_bits_by_fixed_axis": prediction,
            "heldout_actual_bits_by_fixed_axis": actual,
            "prediction_rank_reverses_on_declared_shift": True,
            "fitted_decoder_error_equal_Bayes_error": error,
            "actual_feature_MI_bits": actual_mi, "fitted_decoder_logloss_bits": ce,
            "valid_expected_logloss_information_lower_bound_bits": 1-ce,
            "valid_expected_error_information_lower_bound_bits": fano_lower,
            "scope": "independent stress fixture; no physical-source or benchmark-outcome tuning"}


def rank_one_checks():
    q, k = 9, 3
    basis, _ = np.linalg.qr(rng.normal(size=(q, 5)))
    temp = rng.normal(size=(q, q))
    base = temp @ temp.T/q + np.eye(q)
    chol = np.linalg.cholesky(base)
    white = np.linalg.inv(chol)
    common = rng.normal(size=q)
    plus = common + rng.normal(size=(k, 3)) @ basis[:, :3].T @ chol.T
    minus = common + rng.normal(size=(k, 3)) @ basis[:, :3].T @ chol.T
    backgrounds = rng.normal(size=(k, 2)) @ basis[:, 3:].T @ chol.T
    covariances = np.array([base + np.outer(u, u) for u in backgrounds])
    ref = plus[0]
    rows = np.vstack([(np.vstack([plus, minus])-ref) @ white.T, backgrounds @ white.T])
    _, singular, vt = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-12))
    reduced_map = vt[:rank] @ white
    x = rng.normal(size=(512, q)) @ chol.T + common

    def varying_posterior(samples, pmeans, mmeans, covs):
        def logdensity(mu, covariance):
            diff = samples-mu
            return -.5*(np.einsum("ni,ij,nj->n", diff, np.linalg.inv(covariance), diff)
                        + np.linalg.slogdet(covariance)[1])
        lp = np.column_stack([logdensity(pmeans[h], covs[h]) for h in range(k)])
        lm = np.column_stack([logdensity(mmeans[h], covs[h]) for h in range(k)])
        return expit(logsumexp(lp, axis=1)-logsumexp(lm, axis=1))

    full_p = varying_posterior(x, plus, minus, covariances)
    small_p = varying_posterior((x-ref) @ reduced_map.T, (plus-ref) @ reduced_map.T,
                               (minus-ref) @ reduced_map.T,
                               np.array([reduced_map @ c @ reduced_map.T for c in covariances]))
    projection, _ = np.linalg.qr(rng.normal(size=(q, 4)))
    projection = projection.T
    push_errors, inverse_errors, logdet_errors = [], [], []
    for h, u in enumerate(backgrounds):
        b = white @ u
        expected_inverse = white.T @ (np.eye(q)-np.outer(b, b)/(1+b@b)) @ white
        inverse_errors.append(float(np.linalg.norm(expected_inverse-np.linalg.inv(covariances[h]))))
        logdet_errors.append(abs(float(np.linalg.slogdet(covariances[h])[1]-np.linalg.slogdet(base)[1]-np.log1p(b@b))))
        pu = projection @ u
        push_errors.append(float(np.linalg.norm(projection @ covariances[h] @ projection.T
                           - projection @ base @ projection.T - np.outer(pu, pu))))
    assert rank == 5 and np.max(np.abs(full_p-small_p)) < 1e-12
    assert max(inverse_errors+logdet_errors+push_errors) < 1e-12
    return {"original_dimension": q, "affine_mean_and_background_span_rank": rank,
            "posterior_max_error": float(np.max(np.abs(full_p-small_p))),
            "rank_one_inverse_max_error": max(inverse_errors),
            "rank_one_logdet_max_error": max(logdet_errors),
            "projected_covariance_max_error": max(push_errors),
            "model": "state-static means; independent epoch rank-one Gaussian background"}


def empirical_bernstein_checks():
    alpha, width = .05, 2.
    logfactor = np.log(4/alpha)
    failures = []
    for n in [2, 10, 100, 1000]:
        count = np.arange(n+1)
        mean = -1+width*count/n
        variance = width**2*count*(n-count)/(n*(n-1))
        half = np.sqrt(2*variance*logfactor/n)+7*width*logfactor/(3*(n-1))
        for p in [0., .001, .05, .3, .5, .95, .999, 1.]:
            failure = float(binom.pmf(count, n, p)[np.abs(mean-(-1+width*p)) > half].sum())
            failures.append({"n": n, "p": p, "exact_binomial_failure_probability": failure})
    values = rng.uniform(-1, 1, size=23)
    pair_variance = np.sum((values[:, None]-values[None, :])**2)/(2*len(values)*(len(values)-1))
    assert abs(pair_variance-np.var(values, ddof=1)) < 1e-14
    assert max(row["exact_binomial_failure_probability"] for row in failures) <= alpha+1e-12
    return {"source": "Maurer-Pontil 2009 Theorem 4, both tails with alpha/2 each",
            "url": "https://www.cs.mcgill.ca/~colt2009/papers/012.pdf",
            "two_sided_formula": "sqrt(2*s2*log(4/alpha)/n)+7*width*log(4/alpha)/(3*(n-1)); s2 uses ddof=1",
            "sample_variance_identity_error": float(abs(pair_variance-np.var(values, ddof=1))),
            "checked_binomial_cases": len(failures),
            "maximum_exact_failure_probability": max(row["exact_binomial_failure_probability"] for row in failures),
            "scope": "implementation constants/scaling check, not a replacement for the cited proof"}


def main():
    evidence = {"seed": SEED, "scope": "independent prospective ceiling benchmark mathematical audit",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "raw_CAR_averaging": car_and_averaging(),
                "unequal_midpoint_common_covariance_mixture": midpoint_mixture(),
                "development_prediction_and_fitted_decoder": predictions_and_decoder(),
                "head_dependent_epoch_background": rank_one_checks(),
                "empirical_Bernstein": empirical_bernstein_checks(),
                "reviewed_config_sha256": hashlib.sha256((ROOT/"config/ceiling_benchmark.json").read_bytes()).hexdigest(),
                "reviewed_protocol_sha256": hashlib.sha256((ROOT/"CEILING_BENCHMARK_PROTOCOL.md").read_bytes()).hexdigest(),
                "limitations": "Algebra checked in floating point; MC confidence is pointwise conditional on toy law; quadrature convergence is numerical, not interval-certified; no new theorem or physical outcome claim."}
    out = ROOT / "evidence" / "ceiling_benchmark_math_checks.json"
    out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
