using CSV, DataFrames, Query, Blink, Interact, StatsPlots, DelimitedFiles
using TimeSeriesResampler: resample, mean, ohlc, sum, TimeFrame
using TimeSeries, Distributions, FreqTables, ScikitLearn, LaTeXStrings,LsqFit

function slope(input::AbstractVector{<:Number}, step_size::Int64)
    [i == 1 ? missing : (input[i] - input[i - step_size]) / step_size for i in eachindex(input)]
end

@. gaussian(x,p)=(1/sqrt(2*π*p^2))*exp(((x)^2)/(-2*p^2))

function msd_fit(t,η)
    return t.^η
end

# assets = ["SPX Index","INDU Index","CCMP Index", "UKX Index","DAX Index","NKY Index", "SENSEX Index","NIFTY Index","SPTSX Index","AS51 Index", "CAC Index","SHCOMP Index"]
# assets = ["SPX Index","CCMP Index", "UKX Index"]
exprange(x1, x2, n) = (exp(1)^y for y in range(log(x1), log(x2), length=n))
assets = ["UKX Index"]
intervals=collect(Int.(ceil.(exprange(10,70,6))))
# intervals = [640, 5120, 40960, 163840]

for asset in assets
    csv_obj = CSV.File("./data/$(asset).csv", header=false)
    df = DataFrame(csv_obj)
    df = select(df, :Column1 => :Time, :Column3 => :Price)
    sort!(df, :Time)
    df = dropmissing(df)
    df= DataFrame(df)
    ta = TimeSeries.TimeArray(df, timestamp=:Time)
    big_df = DataFrame()
    fit_df=DataFrame()
    for interval in intervals
        tf = TimeFrame(dt -> floor(dt, Dates.Second(interval)))
        r_df = ohlc(resample(ta, tf))
        r_df = select(DataFrame(r_df), :timestamp => :Time, :Close => :Price)
        prices=r_df.Price #MSD from here
        if length(prices)>10000
            avg_msd=[]
            samples=floor(Int64,length(prices)/10000)
            for i in 1:samples
                if i==1
                    avg_msd=prices[1:i*10000]
                    avg_msd=avg_msd.-avg_msd[1]
                    avg_msd=(avg_msd).^2
                else
                    new=prices[((i-1)*10000)+1:i*10000]
                    new=new.-new[1]
                    new=new.^2
                    avg_msd+=new
                end
            end
            avg_msd/=samples
        mkpath("./$(asset)")
        writedlm("./$(asset)/$(interval)_msd_$(samples)_samples.csv", avg_msd)
        end
            




        # ta = TimeSeries.TimeArray(df, timestamp=:Time)
        # r_df = DataFrame(percentchange(ta))
        r_df.Price = slope(r_df.Price, 1)
        r_df = dropmissing(r_df)
        CSV.write("./$(asset)/$(interval)_pdf.csv", r_df)

        # ft = freqtable(r_df, :Price)
        # ft = log.(ft)
        # ft = DataFrame(price_change=names(ft)[1], freq=ft.array)
        
        # ft.freq .= ifelse.(eltype.(ft.freq), ft.freq, tryparse.(Float64, ft.freq))
        # ft.price_change .= ifelse.(isvalid(Float64,ft.price_change), ft.price_change, tryparse.(Float64, ft.price_change))
        # ft.freq .= ifelse.(iszero.(ft.freq), missing, ft.freq)
        # ft = ft[ft.freq .!= 0.0,:]

        # ft = dropmissing(ft)
        # ft[!, :freq] = convert.(Float64, ft[:, :freq])
        # ft[!, :price_change] = convert.(Float64, ft[:, :price_change])
        # println(first(ft,5))
        # CSV.write("./$(asset)/$(interval)_pdf.csv", ft)

    #     dist_fit = curve_fit(gaussian,ft.price_change,ft.freq,[0.1])
    #     # d=fit(Normal{Float64},(ft.price_change,ft.freq))
    #     # CSV.write("./distribution_pxchange_$(asset)_$(interval).csv", ft)
    #     f_df=DataFrame()
    #     price_change=range(minimum(ft.price_change),maximum(ft.price_change),length=100)
    #     price_change=range(-0.2,0.2,length=100)
    #     ydata=gaussian(price_change,dist_fit.param)
    #     f_df.price_change=price_change
    #     f_df.ydata=ydata
    # gr(size=(800, 600))

    #     plot(price_change,ydata,  label=["fit"], legend=:bottomleft, linewidth=3)
    #     scatter!(ft.price_change, ft.freq, xlabel="Δx Price Change", ylabel=L"P_{\Delta t}(\Delta x)", label=["Δt=$(interval)"], legend=:bottomright, title="Probability dist. for time delays Δt")
    #     mkpath("./plots")
    # isfile("./plots/$(today())_$(asset)_$(interval).png") ?  savefig("./plots/$(today())_$(asset)_$(interval)_$(rand()).png") : savefig("./plots/$(today())_$(asset)_$(interval).png")
    # closeall()
    #     if interval == 640
    #         big_df = ft
    #         fit_df = f_df
    #     else
    #         big_df = outerjoin(DataFrame(big_df), DataFrame(ft), on=:price_change, validate=(true, true), makeunique=true)
    #         fit_df = outerjoin(DataFrame(fit_df), DataFrame(f_df), on=:price_change, validate=(true, true), makeunique=true)
    #     end
    end
    # sort!(big_df, :price_change)
    # sort!(fit_df, :price_change)

    # gr(size=(800, 600))
    # @df big_df plot(:price_change, [:freq,:freq_1,:freq_2,:freq_3], xlabel="Δx Price Change", ylabel=L"P_{\Delta t}(\Delta x)", label=["Δt=640" "5120" "40960" "163840"], legend=:bottomright, title="Probability dist. for different time delays Δt")
    # mkpath("./plots")
    # isfile("./plots/$(today())_$(asset)_algo.png") ?  savefig("./plots/$(today())_$(asset)_algo_$(rand()).png") : savefig("./plots/$(today())_$(asset)_algo.png")
    # closeall()

    # gr(size=(800, 600))
    # @df fit_df plot(:price_change, [:ydata,:ydata_1,:ydata_2,:ydata_3],  label=["Δt=640" "5120" "40960" "163840"], legend=:bottomleft, linewidth=3)
    # mkpath("./plots")
    # isfile("./plots/$(today())_$(asset)_algo.png") ?  savefig("./plots/$(today())_$(asset)_algo_$(rand()).png") : savefig("./plots/$(today())_$(asset)_algo.png")
    # closeall()
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
