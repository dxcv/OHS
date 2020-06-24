"""
1. load option list, together with underlying and maturity date
2. fetch price data from WindPy if not exist in cache
3. calculate Greeks using some Pricing Model(PM)
4. simulate hedge using some hedging strategy(HS), calculate running heding PnL
5. present and report
"""