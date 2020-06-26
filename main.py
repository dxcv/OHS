"""
two usages:
1. exotic hedging strategy research
2. volatility portfolio management
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

    # pw_mat: (inst count + 1, bars, 1)
    pw_mat = -1 * np.ones((d_op_df.shape[1] + 1, d_op_df.shape[0], 1), dtype=float)
    pw_mat[-1, :, 0] = 0.0
    rsk_tgt_mat = np.zeros((d_op_df.shape[0], 3), dtype=float)
    w_df, hr_df = calc_hedge_w_n_expo(op_iv_dct, u_op_df, pw_mat, rsk_tgt_mat, range(87))
    pw_pnl_df, hg_pnl_df = calc_portfolio_pnl(w_df, d_op_df, u_op_df, pw_mat)

    print(1)