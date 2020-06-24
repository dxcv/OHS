from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as bsm_iv
from py_vollib.black_scholes_merton.greeks.analytical import delta as bsm_delta
from py_vollib.black_scholes_merton.greeks.analytical import gamma as bsm_gamma
from py_vollib.black_scholes_merton.greeks.analytical import rho as bsm_rho
from py_vollib.black_scholes_merton.greeks.analytical import theta as bsm_theta
from py_vollib.black_scholes_merton.greeks.analytical import vega as bsm_vega

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


def gen_p_iv_gks_dct(d_tkr_lst, d_p_df, u_p_df, d_tkr2info):
    d_p_dct = {}

    for d in d_tkr_lst:
        tp, K = d_tkr2info[d].values()
        ud_tkr = d_df["u_code"][0]
        tdf = d_p_df.loc[:, [d, "tao", "r", "q"]].join(u_p_df[ud_tkr], how="left")
        tdf["tp"] = tp
        tdf["K"] = K
        tdf["iv"] = sigma = _bsm_iv(tdf[d].values, S := tdf[ud_tkr].values, K,
                            tao := tdf["tao"].values, r := tdf["r"].values, q := tdf["q"].values, tp)

        tdf.columns = ["d_p", "tao", "r", "q", "u_p", "tp", "K", "iv"]

        tdf["delta"] = _bsm_delta(tp, S, K, tao, r, sigma, q)
        tdf["gamma"] = _bsm_gamma(tp, S, K, tao, r, sigma, q)
        tdf["rho"] = _bsm_rho(tp, S, K, tao, r, sigma, q)
        tdf["theta"] = _bsm_theta(tp, S, K, tao, r, sigma, q)
        tdf["vega"] = _bsm_vega(tp, S, K, tao, r, sigma, q)

        d_p_dct[d] = tdf

    return d_p_dct


if __name__ == "__main__":
    from data_preprocsssing import *

    d_df, d_tkr2info = load_derivatives_df_n_cast()
    d_p_df, u_p_df = get_mkt_data(d_df)
    d_p_df = add_date_to_exp(d_p_df)

    p_iv_dct = gen_p_iv_gks_dct(d_df["d_code"], d_p_df, u_p_df, d_tkr2info)
    print(1)