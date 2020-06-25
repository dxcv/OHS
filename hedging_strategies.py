from pandas import DataFrame as pd_DataFrame
from numpy import ndarray as np_ndarray
from numpy import array as np_array
from numpy import random as np_rd
from numpy import isnan as np_isnan
from numpy import ones as np_ones
from numpy import zeros as np_zeros
from scipy.optimize import basinhopping
from multiprocessing import Pool
from functools import partial
from typing import Dict, Tuple, List

from consts import PF_DF_COLS


def construct_portfolio(p_iv_dct: Dict[str, pd_DataFrame], u_p_df, pw_ar) -> Tuple[np_ndarray, List[str]]:
    udf = u_p_df.copy()
    udf.columns = ["d_p"]
    # TODO: use futures to hedge, where these params may change
    udf["delta"] = 1.0
    udf["gamma"] = udf["rho"] = udf["theta"] = udf["vega"] = 0.0

    return np_array([tdf[PF_DF_COLS].values for tdf in p_iv_dct.values()] + [udf.values]) * pw_ar, \
           list(p_iv_dct.keys()) + u_p_df.columns.to_list()


def delta_neutral_ct(port_mat: np_ndarray) -> float:
    return port_mat[1]**2


def optimize_one_day(port_mat: np_ndarray, op_ct_func, hg_tools=(-1,)) -> np_ndarray:
    def _op_tgt(w):
        tw = np_zeros(w.shape, dtype=float)
        tw[hg_tools] = w[hg_tools]
        return op_ct_func(port_mat.T @ tw)

    w_0 = np_zeros(port_mat.shape[0], dtype=float)
    w_0[hg_tools] = np_rd.randn(len(hg_tools))

    tr = np_zeros(port_mat.shape[0], dtype=float)
    tr[hg_tools] = basinhopping(_op_tgt, w_0).x[hg_tools]
    return tr


def calc_hedge_w_n_expo(op_iv_dct: Dict[str, pd_DataFrame], u_p_df: pd_DataFrame, pw_ar: np_ndarray) \
        -> Tuple[pd_DataFrame, pd_DataFrame]:

    o_cub, cols = construct_portfolio(op_iv_dct, u_p_df, pw_ar)
    o_cub[np_isnan(o_cub)] = 0.0

    pool = Pool()
    w_mat = np_array(pool.map(partial(optimize_one_day, op_ct_func=delta_neutral_ct),
                     [o_cub[:, i, :] for i in range(o_cub.shape[1])]))
    pool.close()
    pool.join()

    hg_rst = (o_cub.swapaxes(0, 1) * w_mat.reshape(*w_mat.shape, 1)).sum(axis=1)

    return pd_DataFrame(data=w_mat, columns=cols, index=u_p_df.index), \
           pd_DataFrame(data=hg_rst, columns=PF_DF_COLS, index=u_p_df.index)


def calc_hedge_pnl(w_df, d_op_df, u_op_df):
    # TODO: add transaction costs
    d_rt_df = d_op_df.diff(1).shift(-1)
    u_rt_df = u_op_df.diff(1).shift(-1)
    return (w_df.iloc[:, :-1] * d_rt_df).join(w_df.iloc[:, [-1]] * u_rt_df)


if __name__ == "__main__":
    from data_preprocsssing import *
    from pricing_models import gen_p_iv_gks_dct

    d_df, d_tkr2info = load_derivatives_df_n_cast()
    d_op_df, u_op_df, d_cp_df, u_cp_df = get_mkt_data(d_df)
    d_t_df = add_info2p_df(d_op_df)

    op_iv_dct = gen_p_iv_gks_dct(d_df["d_code"], d_t_df, u_op_df, d_tkr2info)

    pw_ar = np_ones((d_df.shape[0] + 1, 1, 1), dtype=float)
    w_df, hr_df = calc_hedge_w_n_expo(op_iv_dct, u_op_df, pw_ar)
    hpnl_df = calc_hedge_pnl(w_df, d_op_df, u_op_df)

    hpnl_df.fillna(0).sum(1).cumsum().plot()

    print(1)
