from pandas import read_excel as pd_read_excel
from pandas import DatetimeIndex as pd_DatetimeIndex
from pandas import DataFrame as pd_DataFrame
from pandas import read_hdf as pd_read_hdf
from pandas import to_datetime as pd_to_datetime

from os.path import exists as op_exists

from WindPy import w as w_obj

from consts import OPTION_LIST_FILE, D_DF_COLS, START_DT, END_DT, OBS_DT, MKT_DATA_FILE, DAYS_IN_YEAR


def load_derivatives_df_n_cast():
    d_df = pd_read_excel(OPTION_LIST_FILE)
    d_df.columns = D_DF_COLS
    d_df["lst_dt"] = pd_DatetimeIndex(d_df["lst_dt"])
    d_df["type"] = d_df["name"].apply(lambda x: "c" if "购" in x else "p")
    return d_df, d_df.loc[:, ["d_code", "type", "K"]].set_index("d_code").T.to_dict()


def get_mkt_data(d_df):
    if op_exists(MKT_DATA_FILE):
        d_p_df = pd_read_hdf(MKT_DATA_FILE, key="d")
        u_p_df = pd_read_hdf(MKT_DATA_FILE, key="u")
    else:
        w_obj.start()
        data = w_obj.wsd(",".join(d_df["d_code"].to_list()), "close", START_DT, END_DT, "")
        d_p_df = pd_DataFrame(data=data.Data, columns=pd_DatetimeIndex(data.Times), index=data.Codes).T

        data = w_obj.wsd(",".join(d_df["u_code"].unique().tolist()), "close", START_DT, END_DT, "")
        u_p_df = pd_DataFrame(data=data.Data, columns=pd_DatetimeIndex(data.Times), index=data.Codes).T

        w_obj.close()

        d_p_df.sort_index(inplace=True)
        d_p_df.to_hdf(MKT_DATA_FILE, key="d")
        u_p_df.sort_index(inplace=True)
        u_p_df.to_hdf(MKT_DATA_FILE, key="u")

    return d_p_df, u_p_df


def add_date_to_exp(d_p_df):
    d_p_df["tao"] = (pd_to_datetime(OBS_DT) - d_p_df.index).days / DAYS_IN_YEAR

    # TODO: add term structure, or real financing costs
    d_p_df["r"] = 0.03
    d_p_df["q"] = 0.15

    return d_p_df


if __name__ == "__main__":
    d_df, d_tkr2info = load_derivatives_df_n_cast()
    d_p_df, u_p_df = get_mkt_data(d_df)
    d_p_df = add_date_to_exp(d_p_df)
    print(1)
