from pandas import DataFrame as pd_DataFrame
from numpy import ndarray as np_ndarray
from numpy import array as np_array
from numpy import random as np_rd
from numpy import isnan as np_isnan
from numpy import ones as np_ones
from numpy import zeros as np_zeros
from scipy.optimize import least_squares
from multiprocessing import Pool
from functools import partial
from typing import Dict, Tuple, List

from consts import PF_DF_COLS


def gen_risk_metrics(p_iv_dct: Dict[str, pd_DataFrame], u_p_df) -> Tuple[np_ndarray, List[str]]:
    udf = u_p_df.copy()
    udf.columns = ["d_p"]
    # TODO: use futures to hedge, where these params may change
    udf["delta"] = 1.0
    udf["gamma"] = udf["rho"] = udf["theta"] = udf["vega"] = 0.0

    return np_array([tdf[PF_DF_COLS].values for tdf in p_iv_dct.values()] + [udf.values]), \
           list(p_iv_dct.keys()) + u_p_df.columns.to_list()


def delta_neutral_ct(port_mat: np_ndarray) -> float:
    return port_mat[[1, 2, 5]]  # delta, gamma, vega


def optimize_one_day(port_mat: np_ndarray, op_ct_func, hg_tools=(-1,)) -> np_ndarray:
    hg_tools = list(hg_tools)

    def _op_tgt(w):
        tw = np_ones(w.shape, dtype=float)
        tw[hg_tools] = w[hg_tools]
        return op_ct_func(port_mat.T @ tw)

    w_0 = np_ones(port_mat.shape[0], dtype=float)
    w_0[hg_tools] = np_rd.randn(len(hg_tools))

    tr = np_zeros(port_mat.shape[0], dtype=float)
    tr[hg_tools] = least_squares(_op_tgt, w_0).x[hg_tools]
    return tr


def calc_hedge_w_n_expo(op_iv_dct: Dict[str, pd_DataFrame], u_p_df: pd_DataFrame, pw_ar: np_ndarray, hg_tools=(-1,)) \
        -> Tuple[pd_DataFrame, pd_DataFrame]:
    o_cub, cols = gen_risk_metrics(op_iv_dct, u_p_df)
    o_cub = o_cub.astype(float)
    o_cub[np_isnan(o_cub)] = 0.0

    pool = Pool()
    hw_mat = np_array(pool.map(partial(optimize_one_day, op_ct_func=delta_neutral_ct, hg_tools=hg_tools),
                     [o_cub[:, i, :] for i in range(o_cub.shape[1])]))
    pool.close()
    pool.join()

    hg_rst = (o_cub.swapaxes(0, 1) * (pw_ar[:, :, 0] + hw_mat.reshape(*hw_mat.shape, 1))).sum(axis=1)

    return pd_DataFrame(data=hw_mat, columns=cols, index=u_p_df.index), \
           pd_DataFrame(data=hg_rst, columns=PF_DF_COLS, index=u_p_df.index)


def calc_portfolio_pnl(w_df: pd_DataFrame, d_op_df: pd_DataFrame, u_op_df: pd_DataFrame, pw_ar: np_ndarray) \
        -> Tuple[pd_DataFrame, pd_DataFrame]:
    # TODO: add transaction costs
    d_rt_df = d_op_df.diff(1).shift(-1)
    u_rt_df = u_op_df.diff(1).shift(-1)
    return (pw_ar[:-1, 0, 0] * d_rt_df).join(pw_ar[-1, 0, 0] * u_rt_df).sum(1), \
           (w_df.iloc[:, :-1] * d_rt_df).join(w_df.iloc[:, [-1]] * u_rt_df).sum(1)

