using CSV, DataFrames, Query, Blink, Interact, StatsPlots
using TimeSeriesResampler: resample, mean, ohlc, sum, TimeFrame
using TimeSeries, Distributions, FreqTables, ScikitLearn

function slope(input::AbstractVector{<:Number}, step_size::Int64)
    [i == 1 ? missing : (input[i] - input[i - step_size]) / step_size for i in eachindex(input)]
end
# assets = ["SPX Index","INDU Index","CCMP Index", "UKX Index","DAX Index","NKY Index", "SENSEX Index","NIFTY Index","SPTSX Index","AS51 Index", "CAC Index","SHCOMP Index"]
# assets = ["SPX Index","CCMP Index", "UKX Index"]
exprange(x1, x2, n) = (exp(1)^y for y in range(log(x1), log(x2), length=n))
assets = ["USDEUR Curncy"]
# intervals=collect(Int.(ceil.(exprange(640,1600,4))))
intervals = [640,5120,40960,163840]

for asset in assets
    csv_obj = CSV.File("./Work/$(asset).csv", header=false)
    df = DataFrame(csv_obj)
    df = select(df, :Column1 => :Time, :Column3 => :Price)
    sort!(df, :Time)
    df = dropmissing(df)
    ta = TimeSeries.TimeArray(df, timestamp=:Time)
    big_df = DataFrame()
    for interval in intervals
        tf = TimeFrame(dt -> floor(dt, Dates.Second(interval)))
        r_df = ohlc(resample(ta, tf))
        r_df = select(DataFrame(r_df), :timestamp => :Time, :Close => :Price)
        # ta = TimeSeries.TimeArray(df, timestamp=:Time)
        # r_df = DataFrame(percentchange(ta))
        r_df.Price = slope(r_df.Price, 1)
        r_df = dropmissing(r_df)
        # CSV.write("./Work/resampled_$(asset)_$(interval).csv", r_df)
        
        ft = freqtable(r_df, :Price)
        ft = log.(ft)
        ft = DataFrame(price_change=names(ft)[1], freq=ft.array)
        
        # ft.freq .= ifelse.(eltype.(ft.freq), ft.freq, tryparse.(Float64, ft.freq))
        # ft.price_change .= ifelse.(isvalid(Float64,ft.price_change), ft.price_change, tryparse.(Float64, ft.price_change))
        # ft.freq .= ifelse.(iszero.(ft.freq), missing, ft.freq)
        ft=ft[ft.freq .!=0.0,:]

        # ft = dropmissing(ft)
        ft[!, :freq] = convert.(Float64, ft[:, :freq])
        ft[!, :price_change] = convert.(Float64, ft[:, :price_change])
        # CSV.write("./Work/distribution_pxchange_$(asset)_$(interval).csv", ft)
        if interval == 640
            big_df = ft
        else
            big_df = outerjoin(DataFrame(big_df), DataFrame(ft), on=:price_change, validate=(true, true), makeunique=true)
        end
    end
    sort!(big_df, :price_change)

    gr(size=(800,600))
    @df big_df plot(:price_change, [:freq,:freq_1,:freq_2,:freq_3], xlabel="Δx Price Change", ylabel="P_{Δt}(Δx)", label=["Δt=640" "5120" "40960" "163840"], legend=:bottomright, title="Probability dist. for different time delays Δt", linewidth=3)
    mkpath("./Work/plots")
    isfile("./Work/plots/$(today())_$(asset)_algo.png") ?  savefig("./Work/plots/$(today())_$(asset)_algo_$(rand()).png") : savefig("./Work/plots/$(today())_$(asset)_algo.png")
    closeall()
end

# csv_obj=CSV.File("./per_index_parameters_results/$(index)_results.csv")
# df=DataFrame(csv_obj)
# unprofitable_cases = @from i in df begin
#     @where i.CumHoldReturn < 0 && i.Diff_returns<0
#     @select {i.index_name, i.sell_threshold, i.keep_threshold, i.increment, i.CumHoldReturn, i.Diff_returns}
#     @collect DataFrame
# end
# profitable_cases = @from i in df begin
#     @where i.CumHoldReturn > 0 && i.Diff_returns>0
#     @select {i.index_name, i.sell_threshold, i.keep_threshold, i.increment, i.CumHoldReturn, i.Diff_returns}
#     @collect DataFrame
# end
# neutral_cases = @from i in df begin
#     @where i.CumHoldReturn == 0 && i.Diff_returns==0
#     @select {i.drop_threshold, i.rise_threshold, i.train_window, i.CumHoldReturn, i.Diff_returns}
#     @collect DataFrame
# end

# describe(profitable_cases, :min, :max, :mean, :median)
# w = Window()
# body!(w, dataviewer(unprofitable_cases))
