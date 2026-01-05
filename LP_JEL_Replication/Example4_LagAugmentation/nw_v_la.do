***********************************************
* Example to show coverage of error bands
* Newey-West vs lag-augmentation
***********************************************
cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example4_LagAugmentation/"

* graph settings
set more off
set scheme s1color

graph set window fontface "Palatino"
graph set window fontfaceserif "Palatino"
 
gr drop _all

clear
set more off 
set trace off

*** Choose the features of your model

global nobs 300
global burn	1000
global tobs = $nobs + $burn

global ayy	0.85
global ayx	0.2
global axy	0.2
global axx	0.85

global byx	1

global p 	0.05

*** Choose the horizon for your LP
local h		13

set obs  $tobs
gen t = _n

tsset t
*** Generate data from the model
set seed 12345
gen uy 	= rnormal()
gen ux  = rnormal()

gen y 	= 0
gen x 	= 0

replace y = $ayy*l.y + $ayx*l.x + $byx*ux + uy  if _n > 1
replace x = $axy*l.y + $axx*l.x + ux			if _n > 1

drop if _n <= $burn
replace t = _n 

*** Estimate the response of x to a shock in y with LP
*** Generate empty vector to fill with estimates
gen b 	= .
gen snw	= .
gen sla	= .

sort t, stable
tsset t


*** Generate the forward variables and reg
gen h=_n-1

forvalues i = 0/`h'{
	newey f`i'.x y l.y l.x, lag(6)
	replace b 	= _b[y] 	if h == `i'
	replace snw	= _se[y] 	if h == `i'
	
	reg f`i'.x y l(1/2).y l(1/2).x, vce(hc3)
	replace sla	= _se[y] 	if h == `i'
	
}

*** Compare the error bands
gen unw		=	b + invnormal(1 -$p/2)*snw
gen dnw		=	b - invnormal(1 -$p/2)*snw

gen ula		=	b + invnormal(1 -$p/2)*sla
gen dla		=	b - invnormal(1 -$p/2)*sla

gen zero	= 0 



*** Generate the Figure
twoway ///
	(rarea unw dnw  h ,  fc(blue%30) lw(none)) ///
	///(rarea ula dla  h ,  fc(red%20) lw(none)) ///
	(line ula h, lc(pink) lp(dash) lw(medthick)) ///
	(line dla h, lc(pink) lp(dash) lw(medthick)) ///
	(line b h, lc(blue) lp(solid) lw(thick)) /// 
	/// (line zero t, lc(black) lp(dot)) ///
	if h < `h' , ///
	yline(0,lc(black)) ///
	ytit("Response, {it:{&beta}}{subscript:{it:h}}") xtit("Horizon, {it:h}") ///
	legend(row(1) order(- "Response:" 4 "" - -  "Confidence intervals:" 1 "Newey-West" 2 "Lag-augmentated" ) ///
		region(lstyle(none)) size(medium) ) ///
	xlab(0(2)12) ysc(range(0 0.8)) ylab(0(0.2)0.8) ///
	/// note("Sample size: $nobs") ///
	graphregion(color(white)) plotregion(color(white)) ///
	xsize(6) ysize(3)

gr export "fig_nw_v_la.pdf", replace

