*******************************************************
*	KITAGAWA-OAXACA-BLINDER decomposition example
*******************************************************

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/LP_JEL_Replication/Example9_StateDependence"

cap cd "/Users/amtaylor/School of Management Dropbox/Alan M. Taylor/JordaTaylorDropbox/JEL_LP/LP_JEL_Replication/Example9_StateDependence/"

clear
set more off 

set scheme s1color
graph set window fontface "Palatino"


*** Sample size and burn in 
global nobs 500
global burn 500
global tobs = $nobs + $burn

*** Parameters
global rhox 	0.75
global rhos 	0.75
global rhoy		0.75
global mu0  	0
global beta 	0.5
global theta	0.5
global gamma0   0.75


*** LP horizon
local  h		12

*** Start the simulation
set obs  $tobs
gen t = _n

sort t, stable
tsset t
*** Generate data from the model
set seed 12345
gen vy 	= rnormal()
gen vx  = rnormal()
gen vs  = rnormal()


gen y 	= 0
gen x 	= 0
gen s 	= 0


replace x = $rhox*l.x + vx			if _n > 1
replace s = $rhos*l.s + vs			if _n > 1

gen I = (abs(s) > 1)

replace y = $mu0 + $rhoy*l.y + $gamma0*x + ///
I*s*($beta + $theta*x) + vy if _n > 1

*** Drop the burn-in sample
drop if _n <= $burn
replace t = _n 
tsset t
sum s if I==1

*** generate the products
gen Is  = I*s 
gen Isx = I*s*x


*** generate empty vectors to store estimates

gen b 		= .
gen gam 	= .
gen thet 	= .


gen sb		= .
gen sgam	= .
gen sthet	= .


sort t, stable


*** Estimate the LP
qui{
forvalues i = 0/`h'{
	newey f`i'.y x l.y Is Isx, lag(`h')
	replace b 	 = _b[Is] 	 if _n == `i'
	replace gam  = _b[x]     if _n == `i'
	replace thet = _b[Isx]	 if _n == `i'


	replace sb 	  = _se[Is] 	 if _n == `i'
	replace sgam  = _se[x]       if _n == `i'
	replace sthet = _se[Isx]	 if _n == `i'
	}

forvalues i=1/`h'{
	gen betx_`i' = b[`i'] + thet[`i']*x 
	label var betx_`i' "R(`i')"
}
}	
/*
preserve
keep betx_* //if t > 950
outsheet betx_* using bet_data.csv if _n > 950, comma replace
save bet_data, replace
restore
*/

gen zero = 0



*** Generate the Figure
	twoway (line betx_1 t, lcolor(blue%50) ///
	lpattern(solid) lwidth(medthick)) /// 
	(line betx_3 t, lcolor(green%50) ///
	lpattern(solid) lwidth(medium)) /// 
	(line betx_5 t, lcolor(purple%50) ///
	lpattern(dash) lwidth(medium)) /// 
	(line betx_7 t, lcolor(orange%50) ///
	lpattern(dash) lwidth(medium)) /// 
	(line zero t, lcolor(black)) in 1/100, ///
	title("Panel (a): Time series response variation", ///
	color(black) size(medium)) ///
	xtitle("Observation, {it:t}", size(medsmall)) ///
	ytitle("Response, state-varying, {fontface cmsy10: R}({it:h})", size(medsmall)) ///
	graphregion(color(white)) plotregion(color(white)) ///
	legend(order( ///
		1 "{fontface cmsy10: R}(0)" ///
		2 "{fontface cmsy10: R}(2)" ///
		3 "{fontface cmsy10: R}(4)" ///
		4 "{fontface cmsy10: R}(6)" ///
		) region(lstyle(none))) ///
	name(tskob, replace)  xsize(3) ysize(3.5)
	gr rename g1, replace

*** Generate the Figure w/ no title
	twoway (line betx_1 t, lcolor(blue%50) ///
	lpattern(solid) lwidth(medthick)) /// 
	(line betx_3 t, lcolor(green%50) ///
	lpattern(solid) lwidth(medium)) /// 
	(line betx_5 t, lcolor(purple%50) ///
	lpattern(dash) lwidth(medium)) /// 
	(line betx_7 t, lcolor(orange%50) ///
	lpattern(dash) lwidth(medium)) /// 
	(line zero t, lcolor(black)) in 1/100, ///
	///title("Panel (a): Time series response variation", ///
	///color(black) size(medium)) ///
	xtitle("Observation, {it:t}", size(medsmall)) ///
	ytitle("Response, state-varying, {fontface cmsy10: R}({it:h})", size(medsmall)) ///
	graphregion(color(white)) plotregion(color(white)) ///
	legend(order( ///
		1 "{fontface cmsy10: R}(0)" ///
		2 "{fontface cmsy10: R}(2)" ///
		3 "{fontface cmsy10: R}(4)" ///
		4 "{fontface cmsy10: R}(6)" ///
		) region(lstyle(none))) ///
	name(tskob, replace)  xsize(3) ysize(3.5)
	gr rename g1nt, replace

	gr export "fig_kob_1nt.pdf", replace

drop if t > `h'



gen bu1 = b + thet
gen bu2 = b + 2*thet
gen bd1 = b - thet
gen bd2 = b -2*thet

gen sbu = b + 1.96*sb
gen sbd = b - 1.96*sb



*** Generate the Figure
	twoway (rarea sbu sbd  t, ///
	fcolor(blue%20) lcolor(white) lpattern(solid)) ///
	(line b t, lcolor(gray) ///
	lpattern(solid) lwidth(thick)) /// 
	(line bu1 t, lcolor(blue%75) ///
	lpattern(dash) lwidth(thick)) /// 
	(line bd1 t, lcolor(purple%75) ///
	lpattern(dash) lwidth(thick)) /// 
	(line bu2 t, lcolor(blue%50) ///
	lpattern(dash) lwidth(medthick)) /// 
	(line bd2 t, lcolor(purple%50) ///
	lpattern(dash) lwidth(medthick)) /// 
	(line zero t, lcolor(black)), ///
	title("Panel (b): Variation due to secondary treatment", ///
	color(black) size(medium)) ///
	xtitle("Horizon, {it:h}", size(medsmall)) ///
	ytitle("Response, state-varying, {fontface cmsy10: R}({it:h})", size(medsmall)) ///
	xlabel(2 4 6 8 10 12	) ///
	xscale(extend) ///
	graphregion(color(white)) plotregion(color(white)) ///
	legend(order(3 "2{superscript:nd} Treat =  1" 5 "2{superscript:nd} Treat =  2" ///
	4 "2{superscript:nd} Treat = -1" 6 "2{superscript:nd} Treat = -2") ///
	region(lstyle(none)))  xsize(3) ysize(3.5)
	gr rename g2, replace
	
*** Generate the Figure w/ no title
	twoway (rarea sbu sbd  t, ///
	fcolor(blue%20) lcolor(white) lpattern(solid)) ///
	(line b t, lcolor(gray) ///
	lpattern(solid) lwidth(thick)) /// 
	(line bu1 t, lcolor(blue%75) ///
	lpattern(dash) lwidth(thick)) /// 
	(line bd1 t, lcolor(purple%75) ///
	lpattern(dash) lwidth(thick)) /// 
	(line bu2 t, lcolor(blue%50) ///
	lpattern(dash) lwidth(medthick)) /// 
	(line bd2 t, lcolor(purple%50) ///
	lpattern(dash) lwidth(medthick)) /// 
	(line zero t, lcolor(black)), ///
	///title("Panel (b): Variation due to secondary treatment", ///
	///color(black) size(medium)) ///
	xtitle("Horizon, {it:h}", size(medsmall)) ///
	ytitle("Response, state-varying, {fontface cmsy10: R}({it:h})", size(medsmall)) ///
	xlabel(2 4 6 8 10 12	) ///
	xscale(extend) ///
	graphregion(color(white)) plotregion(color(white)) ///
	legend(order(3 "2{superscript:nd} Treat =  1" 5 "2{superscript:nd} Treat =  2" ///
	4 "2{superscript:nd} Treat = -1" 6 "2{superscript:nd} Treat = -2") ///
	region(lstyle(none))) xsize(3) ysize(3.5)
	gr rename g2nt, replace
	
	gr export "fig_kob_2nt.pdf", replace


	gr combine g1 g2, ycommon  ///
	graphregion(color(white)) plotregion(color(white))
	

	graph save   fig_kob_combi.gph, replace
	gr export "fig_kob_combi.pdf", replace

	* clean up and save for use in the paper

copy fig_kob_1nt.pdf  		Figure10a.pdf , replace
copy fig_kob_2nt.pdf  		Figure10b.pdf , replace

