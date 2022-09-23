L:{x0:.Q.s[x]; x0[where x0="\""]:" "; x0[where x0="\n"]:" "; -1 "[",(string `time$.z.Z), "] ",x0;}

generate:{ [ND;x0]
	:([] time:2016.01.01 + til ND;
	open:x0+(floor (ND?0.99)*100)%100;
	high:x0+(floor (ND?0.99)*100)%100;
	low:x0+(floor (ND?0.99)*100)%100;
	close:x0+(floor (ND?0.99)*100)%100;
	volume:1000+(floor (ND?10000)*100)%100)
	}

L "Generating testing databases ..."

ND:365
D_MSFT: generate[3*ND; 50]
D_AAPL: generate[ND;   100]
D_GE:   generate[3*ND; 50]
D_AAL:  generate[3*ND; 20]
D_SPY:  generate[3*ND; 190]
D_AAL_DOT_TEST:  generate[3*ND; 20]


L "Done"

/ --- interface functions
i_series:{ :{ :{2 _ (string x)} each x[where {(string x) like "D_*"} each x] }[system "a"] }

i_timeframe:{ :enlist 24*3600 }

/ - retrieve all prices for given instrument in date ranges
i_fetch:{[instr;nBar;start;end]
	:$[nBar<=86400;
		eval parse "select from D_",(upper (string instr))," where (`date$time) within ",(string start)," ",(string end);
		[
		p:floor nBar%86400;
		eval parse "select open:min open,high:max high,low:min low,close:last close,volume:sum volume by ",(string p)," xbar time:`date$time from D_",(upper(string instr))," where (`date$time) within ",(string start)," ",(string end)
		]
	]
	}

