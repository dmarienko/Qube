
gen_tick_day:{[date; N; p0; d0; spread]
	p:p0+d0*floor[100*(sin (1 + til N)%100)]%100;
	:`time xasc ([] time:date+09:30:00.0+N?24000000;
	ask:p;
	bid:p+spread;
	bidvol:(N?10)*100;
	askvol:(N?10)*100)
	}

gen_ticks_days_range:{[dates; N; p0; d0; sprd]
	raze (gen_tick_day[dates[0]; N; p0; d0; sprd] upsert\ gen_tick_day[; N; p0; d0; sprd] each 1 _ dates)
	}

t_msft:gen_ticks_days_range[(2016.01.01 + til 10); 1000; 50; 2; 0.01]
t_xom:gen_ticks_days_range[(2016.01.01 + til 10); 100000; 35; 2; 0.01]
t_aapl:gen_ticks_days_range[(2016.01.01 + til 10); 100000; 90; 3; 0.01]

/ --- interface functions
i_series:{ :{ :{2 _ (string x)} each x[where {(string x) like "t_*"} each x] }[system "a"] }

i_timeframe:{ :enlist 0 }

/ - 5 min for testing
i_fetch:{[symbol;nBar;start;end]
	nBar:300;
	t0:eval parse "select open:first (ask+bid)%2,high:max (ask+bid)%2,low:min (ask+bid)%2,close:last (ask+bid)%2,volume:count ask by ",(string nBar)," xbar time:time.second, date:`date$time from t_",(lower (string symbol))," where time within ",(string start)," ",(string end);
	:select time:date+time,open,high,low,close,volume from t0
	}
