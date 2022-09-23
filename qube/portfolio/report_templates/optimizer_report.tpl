#set($format02 = "{0:.2f}")
#set($format03 = "{0:.3f}")


#if($meta)
<h3 style="text-align:center;">Optimizer #if($meta["name"]) <strong>[$meta["name"]]</strong> #end <strong>$meta["id"]</strong><br>
</h3>
#end

#set($cagr = $data["cagr"])
#set($drawdown_pct = $data["drawdown_pct"])
#set($params = $data["params"])
#set($sharpe = $data["sharpe"])
#set($sortino = $data["sortino"])
#set($mdd_usd = $data["mdd_usd"])
#set($mdd_duration = $data["mdd_duration"])
#set($terminal_pnl = $data["terminal_pnl"])
#set($run_id = $data["optimizer_iteration_run_id"])
<script>
function checkall(){
    var checked = true;
    x = document.getElementsByClassName("checkall_checkbox");
    for (i = 0; i < x.length; i++) {
    if (x[i].checked==false){
        checked = false
    }
    }
    for (i = 0; i < x.length; i++) {
    if (checked==false){
       x[i].checked=true;
    }else{
        x[i].checked=false;
    }
    }

}
</script>
<form method="POST" action="/explore/backtesting/tester/comparison_report" id="comparison_form">
<input type="hidden" name="fluid" value="true">
<button onclick="document.getElementById('comparison_form').submit();" type="button" class="btn btn-primary">Comparison Report</button><br><br>
<table class="sortable" width=100%>
<thead>
<tr>
<th>params</th>
<th>sharpe</th>
<th>cagr</th>
<th>drawdown_pct</th>
<th>sortino</th>
<th>Max dd (USD)</th>
<th>Terminal PnL (USD)</th>
<th>Max DD Duration</th>
<th style="text-align:center;"><span class="checkall" style='font-size:14px; text-decoration:underline; cursor:pointer; color: #243d56' onclick="checkall();">check all</span></th>
</tr>
</thead>
<tbody>
#foreach ($param in $params)
<tr>
<td>$params[$param]</td>
<td>#if($sharpe[$param])$format03.format($sharpe[$param])#end</td>
<td>#if($cagr[$param])$format02.format($cagr[$param])#end</td>
<td>#if($drawdown_pct[$param])$format02.format($drawdown_pct[$param])#end</td>
<td>#if($sortino[$param])$format03.format($sortino[$param])#end</td>
<td>#if($mdd_usd[$param])$format03.format($mdd_usd[$param])#end</td>
<td>#if($terminal_pnl[$param])$format03.format($terminal_pnl[$param])#end</td>
<td>$mdd_duration[$param]</td>
<td style="text-align:center;"><input name="run_ids[]" type="checkbox" class="checkall_checkbox" value="$run_id[$param]"></td>
</tr>
#end
<tbody>
</table>
</form>
<br><br>
<button onclick="document.getElementById('comparison_form').submit();" type="button" class="btn btn-primary">Comparison Report</button>
