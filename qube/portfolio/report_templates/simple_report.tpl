#set($format02 = "{0:.2f}")
#set($format03 = "{0:.3f}")

#set($cagr_percent = $cagr * 100)

#set($annual_percent = $annual_volatility * 100)

#set($return_percent = $mean_return * 100)

<style>
.report_table td, .report_table th {
    text-align:left !important;
}

.wrap_table th {
    text-align:center !important;
}

.wrap_table td, .wrap_table tr {
    background: none !important;
    text-align:left !important;
}

.flex-container {
   display: flex;
  align-items: flex-start;
}

.table_block {
  width:30%;
}

</style>

<table class="wrap_table" width=100%>
<th><font size=3 color="green" text-align="center">#if ($strategy_name) $strategy_name #else Simulation #end</font></th>
<tr><td><font color="#3030ff" size=-2>$simulation_info</font></td></tr>
</table>

<div class="flex-container"><div class="">
<img src='$chart($strategy_name, $equity, $mdd_start, $mdd_recover, $compound_returns, $drawdown_usd, $long_value, $short_value, $insample, $behcmark_compound, $figsize, $fontsize, True)'>
</div>
<div class="table_block">
<table class="report_table" width=100%>
<tr>
     <th> Statistic </th><th> Model </th>
</tr>
<tr><td>Gain (USD):</td><td>$format02.format($gain)</td></tr>
<tr><td>CAGR (%):</td><td>$format03.format($cagr_percent)</td></tr>
<tr><td>Sharpe:</td><td>$format03.format($sharpe)</td></tr>
<tr><td>QR:</td><td>$format03.format($qr)</td></tr>
<tr><td>Sortino:</td><td>$format03.format($sortino)</td></tr>
<tr><td>Calmar: </td><td>$format03.format($calmar)</td></tr>
<tr><td>MaxDD (%):</td><td>$format03.format($drawdown_pct)</td></tr>
<tr><td>MaxDD On Init BP (%):</td><td>$format03.format($drawdown_pct_on_init_bp)</td></tr>
<tr><td>MaxDD (USD):</td><td>$format03.format($mdd_usd)</td></tr>
<tr><td>MaxDD days:</td><td>$max_dd_days.days days</td></tr>
<tr><td>MaxDD start:</td><td>$mdd_start.date()</td></tr>
<tr><td>MaxDD peak:</td><td>$mdd_peak.date()</td></tr>
<tr><td>MaxDD recover:</td><td>$mdd_recover.date()</td></tr>
<tr><td>Tail Ratio:</td><td>$format03.format($tail_ratio)</td></tr>
<tr><td>Stability: </td><td>$format03.format($stability)</td></tr>
<tr><td>Alpha:</td><td>#if($alpha)$format03.format($alpha)#end</td></tr>
<tr><td>Beta:</td><td>#if($beta)$format03.format($beta)#end</td></tr>
<tr><td>Ann.Volatility (%):</td><td>$format02.format($annual_percent)</td></tr>
<tr><td>VAR_95 (USD):</td><td>$format02.format($var)</td></tr>
<tr><td>Mean return (%)</td><td>$format03.format($return_percent)</td></tr>
<tr><td>Signals</td><td>$number_signals</td></tr>
<tr><td>Broker's Commissions:</td><td>#if ($broker_commissions) $format03.format($broker_commissions) #end</td></tr>

#if($meta.parameters)
   #foreach ($k in $meta.parameters) <tr><td><font color="#EA563E">$k</font></td><td>$meta.parameters[$k]</td></tr>
#end#end

</table>
</div>
</div></div>
