"""Full-source ensemble oracle; depends only on PR 1's polynomial oracle."""
from fractions import Fraction as F
from pathlib import Path
import importlib.util
path=Path(__file__).resolve().parent.parent/'hawkes-forecast/oracle.py'
spec=importlib.util.spec_from_file_location('hawkes_forecast_oracle',path)
forecast=importlib.util.module_from_spec(spec);spec.loader.exec_module(forecast)
full_forecast,add,scale,mul,taylor,source_tools=(forecast.full_forecast,forecast.add,forecast.scale,forecast.mul,forecast.taylor,forecast.source_tools)

def group_forecasts(root, source, weighted_rows, center):
    groups = {}
    for weight, row in weighted_rows:
        key, *point = row
        _, _, polynomials = full_forecast(root, source, point, center)
        if key not in groups:
            groups[key] = [0, [[F(0)] for _ in polynomials]]
        groups[key][0] += weight
        groups[key][1] = [add(a, scale(b, weight)) for a, b in zip(groups[key][1], polynomials)]
    out = []
    for key in sorted(groups):
        mass, sums = groups[key]
        assert mass>0
        means = [scale(p, F(1, mass)) for p in sums]
        covariance = add(means[2], scale(mul(means[0], means[1]), -1)) if len(means)==3 else None
        out.append([1, [key, mass, [taylor(p, center) for p in means], taylor(covariance, center) if covariance is not None else 'not-requested']])
    return out
