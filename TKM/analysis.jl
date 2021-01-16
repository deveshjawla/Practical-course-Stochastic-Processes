using CSV, DataFrames, Query, Blink, Interact, StatsPlots
using TimeSeriesResampler: resample, mean, ohlc, sum, TimeFrame
using TimeSeries, Distributions, FreqTables, ScikitLearn

assets = ["SPX Index","CCMP Index", "UKX Index"]
exprange(x1, x2, n) = (exp(1)^y for y in range(log(x1), log(x2), length=n))

intervals=collect(Int.(ceil.(exprange(30,163840,10))))

for asset in assets, interval in intervals
    csv_obj=CSV.File("./$(interval)_seconds_OHLC/$(asset).csv")
    df=DataFrame(csv_obj)
    df=select(df,:timestamp=>:Time,:Close=>:Price)
    ta = TimeSeries.TimeArray(df, timestamp=:Time)
    df_pct=DataFrame(percentchange(ta))
    #CSV.write("./$(interval)_seconds_OHLC/$(asset)_pct_change.csv",df_pct)
    freq_table=freqtable(df_pct,:Price)
end
