#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


from data_preprocsssing import *
from pricing_models import *
from hedging_strategies import *
from volatility_models import WingModel
from utils import pool_init
import numpy as np
import pandas as pd


# In[3]:


d_df, d_tkr2info = load_derivatives_df_n_cast()
d_op_df, u_op_df, d_cp_df, u_cp_df = get_mkt_data_minutes(d_df)
d_t_df = add_info2p_df(d_op_df)

op_iv_dct = gen_p_iv_gks_dct(d_df["d_code"], d_t_df, u_op_df, d_tkr2info)

# pw_mat: (inst count + 1, bars, 1)
# pw_mat = -1 * np.ones((d_op_df.shape[1] + 1, d_op_df.shape[0], 1), dtype=float)
# pw_mat[-1, :, 0] = 0.0
# rsk_tgt_mat = np.zeros((d_op_df.shape[0], 3), dtype=float)
# w_df, hr_df = calc_hedge_w_n_expo(op_iv_dct, u_op_df, pw_mat, rsk_tgt_mat, (-2,-1))
# pw_pnl_df, hg_pnl_df = calc_portfolio_pnl(w_df, d_op_df, u_op_df, pw_mat)


# In[4]:


from tqdm import tqdm


# In[ ]:


mat_dt_lst = d_df.lst_dt.unique().tolist()

df_lst = []
for mdt in mat_dt_lst:
    op_tkrs = d_df[d_df.lst_dt == pd.to_datetime(mdt)].d_code

    mdl = WingModel(2.9, 2.9, vol_ref=0.5)

    rst_lst = []
    for t in tqdm(range(len(d_op_df))):
        tdf = pd.DataFrame(data=[op_iv_dct[tkr].iloc[t, :] for tkr in op_tkrs], index=op_tkrs).loc[:, ["K", "mkt_iv", "tp"]]
        tdf.sort_values("K", inplace=True)

        cdf = tdf[tdf.tp == "c"].dropna()
        pdf = tdf[tdf.tp == "p"].dropna()

        cdf["pos"] = cdf.mkt_iv.values - mdl.fit(cdf.mkt_iv.values, cdf.K.values).predict(cdf.K.values)
        pdf["pos"] = pdf.mkt_iv.values - mdl.fit(pdf.mkt_iv.values, pdf.K.values).predict(pdf.K.values)

        tdf["pos"] = cdf["pos"].append(pdf["pos"])
        rst_lst.append(tdf["pos"])
    
    df = pd.concat(rst_lst, axis=1)
    df_lst.append(df)


# In[ ]:


import pickle

with open("df_lst", "wb") as fp:
    pickle.dump(fp, df_lst)


# In[ ]:




