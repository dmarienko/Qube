L:{x0:.Q.s[x]; x0[where x0="\""]:" "; x0[where x0="\n"]:" "; -1 "[",(string `time$.z.Z), "] ",x0;}

L "Generating testing databases ..."

gen_tick_day:{[date; N; a0; b0]
	:`time xasc ([] time:date+09:30:00.0+N?24000000;
	ask:a0+(floor (N?0.99)*100)%100;
	bid:b0+(floor (N?0.99)*100)%100;
	bidvol:(N?10)*100;
	askvol:(N?10)*100)
	}

gen_ticks_days_range:{[dates; N; a0; b0]
	raze (gen_tick_day[dates[0]; N; a0; b0] upsert\ gen_tick_day[; N; a0; b0] each 1 _ dates)
	}

T_MSFT:gen_ticks_days_range[(2016.01.01 + til 10); 1000; 50.1; 50.0]
T_SPY:gen_ticks_days_range[(2016.01.01 + til 10); 100000; 190.1; 190.0]

L "Done"

/ --- interface functions
i_series:{ :{ :{2 _ (string x)} each x[where {(string x) like "T_*"} each x] }[system "a"] }

i_timeframe:{ :enlist 0 }

/ - retrieve all prices for given instrument in date ranges
i_fetch:{[symbol;nBar;start;end]
	:$[nBar=0; / loading raw ticks
		eval parse "select time, ask, bid, askvol, bidvol from T_",(upper (string symbol))," where time within ",(string start)," ",(string end);
		[ / loading integrated (here we use midprice)
		t0:eval parse "select open:first (ask+bid)%2,high:max (ask+bid)%2,low:min (ask+bid)%2,close:last (ask+bid)%2,volume:count ask by ",(string nBar)," xbar time:time.second, date:`date$time from T_",(upper (string symbol))," where time within ",(string start)," ",(string end);
		select time:date+time,open,high,low,close,volume from t0
		]
	]
	}
