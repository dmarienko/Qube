#set($format02 = "{0:.2f}")
#set($format03 = "{0:.3f}")

#set($cagr_percent = $cagr * 100)

#if($benchmark_cagr)
#set($bench_cagr_percent = $benchmark_cagr * 100)
#end

#if($benchmark_drawdown_pct)
#set($bench_maxdd_percent = $benchmark_drawdown_pct * 100)
#end

#set($annual_percent = $annual_volatility * 100)

#if($benchmark_annual_volatility)
#set($bench_annual_percent = $benchmark_annual_volatility * 100)
#end

#set($return_percent = $mean_return * 100)

#if($benchmark_mean_return)
#set($bench_return_percent = $benchmark_mean_return * 100)
#end

<style>
.report_table td, .report_table th {
    text-align:left !important;
}
</style>

#if($meta)
#if($meta["error"])
<h3> Simulation complete with following errors: </h3>
<pre>
$meta["error"]
</pre>
#end
<h3 style="text-align:center;">Test #if($meta["name"]) <strong>[$meta["name"]]</strong> #end<strong>$meta["id"]</strong><br>
<strong>$meta["instruments_count"]</strong> instruments.
From <strong>$meta["start_time"]</strong> to <strong>$meta["end_time"]</strong>
</h3>
#end

#if($highcharts)
#set($charts = $chart($strategy_name, $equity, $mdd_start, $mdd_recover, $compound_returns, $drawdown_usd, $long_value, $short_value, $behcmark_compound, $figsize, $fontsize))
#foreach ($chart in $charts)
<div id="$chart.name"></div>
<script>
$chart.chart
</script>
#end

<script>
</script>
#else
<img src='$chart($equity, $mdd_start, $mdd_recover, $compound_returns, $drawdown_usd, $long_value, $short_value, $insample, $behcmark_compound, $figsize, $fontsize)'>
#end

<table class="report_table" width=100%>
<tr>
<th>
Statistic
</th>
<th>
Model
</th>
<th>Benchmark</th>
</tr>
<tr><td>Gain (USD):</td><td>$format02.format($gain)</td><td>#if($benchmark_gain)$format02.format($benchmark_gain)#end</td></tr>
<tr><td>CAGR (%):</td><td>$cagr_percent</td><td>$bench_cagr_percent</td></tr>
<tr><td>Sharpe:</td><td>$format03.format($sharpe)</td><td>#if($benchmark_sharpe)$format03.format($benchmark_sharpe)#end</td></tr>
<tr><td>Sortino:</td><td>$format03.format($sortino)</td><td>#if($benchmark_sortino)$format03.format($benchmark_sortino)#end</td></tr>
<tr><td>Calmar: </td><td>$format03.format($calmar)</td><td>#if($benchmark_calmar)$format03.format($benchmark_calmar)#end</td></tr>
<tr><td>MaxDD (%):</td><td>$drawdown_pct</td><td>$bench_maxdd_percent</td></tr>
<tr><td>MaxDD On Init BP (%):</td><td>$drawdown_pct_on_init_bp</td><td></td></tr>
<tr><td>MaxDD (USD):</td><td>$mdd_usd</td><td>#if($benchmark_max_dd_usd)$format02.format($benchmark_max_dd_usd)#end</td></tr>
<tr><td>MaxDD days:</td><td colspan=2>$max_dd_days.days days [$mdd_start.date() ~ $mdd_peak.date() ~ $mdd_recover.date()]</td></tr>
<tr><td>Tail Ratio:</td><td>$format03.format($tail_ratio)</td><td>#if($benchmark_tail_ratio)$format03.format($benchmark_tail_ratio)#end</td></tr>
<tr><td>Stability: </td><td>$format03.format($stability)</td><td>#if($benchmark_stability)$format03.format($benchmark_stability)#end</td></tr>
<tr><td>Alpha:</td><td>#if($alpha)$format03.format($alpha)#end</td><td>0</td></tr>
<tr><td>Beta:</td><td>#if($beta)$format03.format($beta)#end</td><td>1</td></tr>
<tr><td>Ann.Volatility (%):</td><td>$format02.format($annual_percent)</td><td>#if($bench_annual_percent)$format02.format($bench_annual_percent)#end</td></tr>
<tr><td>VAR_95 (USD):</td><td>$format02.format($var)</td><td>#if($benchmark_var)$format02.format($benchmark_var)#end</td></tr>
<tr><td>Mean return (%)</td><td>$return_percent</td><td>$bench_return_percent</td></tr>
<tr><td>Broker's Commissions:</td><td>$broker_commissions</td><td></td></tr>
</table>