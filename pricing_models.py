from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as bsm_iv
from py_vollib.black_scholes_merton.greeks.analytical import delta as bsm_delta
from py_vollib.black_scholes_merton.greeks.analytical import gamma as bsm_gamma
from py_vollib.black_scholes_merton.greeks.analytical import rho as bsm_rho
from py_vollib.black_scholes_merton.greeks.analytical import theta as bsm_theta
from py_vollib.black_scholes_merton.greeks.analytical import vega as bsm_vega

from typing import List, Dict, Tuple
from pandas import DataFrame as pd_DataFrame
from pandas import Series as pd_Series

from utils import _vec, _no_except


@_vec
@_no_except
def _bsm_iv(*args, **kwargs):
    return bsm_iv(*args, **kwargs)


@_vec
@_no_except
def _bsm_delta(*args, **kwargs):
    return bsm_delta(*args, **kwargs)


@_vec
@_no_except
def _bsm_gamma(*args, **kwargs):
    return bsm_gamma(*args, **kwargs)


@_vec
@_no_except
def _bsm_rho(*args, **kwargs):
    return bsm_rho(*args, **kwargs)


@_vec
@_no_except
def _bsm_theta(*args, **kwargs):
    return bsm_theta(*args, **kwargs)


@_vec
@_no_except
def _bsm_vega(*args, **kwargs):
    return bsm_vega(*args, **kwargs)


def iv2md_vol(tdf: pd_DataFrame) -> pd_Series:
    # TODO: use a model to calculate modle iv
    return tdf["mkt_iv"]


def gen_p_iv_gks_dct(d_tkr_lst: List, d_p_df: pd_DataFrame, u_p_df: pd_DataFrame, d_tkr2info: Dict) \
        -> Dict[str, pd_DataFrame]:

    p_iv_dct = {}

    for d in d_tkr_lst:
        tdct = d_tkr2info[d]
        tp, K, ud_tkr = tdct["type"], tdct["K"], tdct["u_code"]
        tdf = d_p_df.loc[:, [d, "tao", "r", "q"]].join(u_p_df[ud_tkr], how="left")
        tdf["tp"] = tp
        tdf["K"] = K
        tdf["mkt_iv"] = _bsm_iv(tdf[d].values, S := tdf[ud_tkr].values, K,
                            tao := tdf["tao"].values, r := tdf["r"].values, q := tdf["q"].values, tp)

        tdf["md_iv"] = sigma = iv2md_vol(tdf)

        tdf.columns = ["d_p", "tao", "r", "q", "u_p", "tp", "K", "mkt_iv", "md_iv"]

        tdf["delta"] = _bsm_delta(tp, S, K, tao, r, sigma, q)
        tdf["gamma"] = _bsm_gamma(tp, S, K, tao, r, sigma, q)
        tdf["rho"] = _bsm_rho(tp, S, K, tao, r, sigma, q)
        tdf["theta"] = _bsm_theta(tp, S, K, tao, r, sigma, q)
        tdf["vega"] = _bsm_vega(tp, S, K, tao, r, sigma, q)

        p_iv_dct[d] = tdf

    return p_iv_dct


def construct_vol_suface(p_iv_dct: Dict[str, pd_DataFrame]) -> Tuple[pd_DataFrame]:
    pass
