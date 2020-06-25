"""
1. load option list, together with underlying and maturity date
2. fetch price data from WindPy if not exist in cache
3. calculate Greeks using some Pricing Model(PM)
4. simulate hedge using some hedging strategy(HS), calculate running heding PnL
5. present and report
"""

from data_preprocsssing import *
from pricing_models import *
from hedging_strategies import *
import numpy as np
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    d_df, d_tkr2info = load_derivatives_df_n_cast()
    d_op_df, u_op_df, d_cp_df, u_cp_df = get_mkt_data_minutes(d_df)
    d_t_df = add_info2p_df(d_op_df)

    op_iv_dct = gen_p_iv_gks_dct(d_df["d_code"], d_t_df, u_op_df, d_tkr2info)

    pw_ar = -1 * np.ones((d_df.shape[0] + 1, 1, 1), dtype=float)
    pw_ar[-1, 0, 0] = 0.0
    w_df, hr_df = calc_hedge_w_n_expo(op_iv_dct, u_op_df, pw_ar, range(87))
    pw_pnl_df, hg_pnl_df = calc_portfolio_pnl(w_df, d_op_df, u_op_df, pw_ar)

    print(1)