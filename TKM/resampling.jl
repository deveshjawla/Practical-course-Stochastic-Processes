using CSV, DataFrames, Query, Blink, Interact, StatsPlots
using TimeSeriesResampler: resample, mean, ohlc, sum, TimeFrame
using TimeSeries

assets = ["SPX Index","INDU Index","CCMP Index", "UKX Index","DAX Index","NKY Index", "SENSEX Index","NIFTY Index","SPTSX Index","AS51 Index", "CAC Index","SHCOMP Index"]
#assets = ["SPX Index","CCMP Index", "UKX Index"]
exprange(x1, x2, n) = (exp(1)^y for y in range(log(x1), log(x2), length=n))

intervals=collect(Int.(ceil.(exprange(30,163840,10))))

for asset in assets, interval in intervals
    csv_obj=CSV.File("./$(asset).csv",header=false)
    df=DataFrame(csv_obj)
    df=select(df,:Column1=>:Time,:Column3=>:Price)
    ta = TimeSeries.TimeArray(df, timestamp=:Time)
    tf = TimeFrame(dt -> floor(dt, Dates.Second(interval)))
    resampled=ohlc(resample(ta, tf))
    mkpath("./$(interval)_seconds_OHLC")
    CSV.write("./$(interval)_seconds_OHLC/$(asset).csv",resampled)
end
