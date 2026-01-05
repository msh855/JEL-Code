****************************************
*** Small sample bias illustration
****************************************

cap cd "/Users/amtaylor/School of Management Dropbox/Alan M. Taylor/JordaTaylorDropbox/JEL_LP/LP_JEL_Replication/Example1_LongDifferences/"

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/LP_JEL_Replication/Example1_LongDifferences"

clear
set more off 

set scheme s1color
graph set window fontface "Palatino"

*** MC Simulation reps
//global nreps 10 // 1000

*** Sample size and burn in 
//global nobs 100 // 300
global burn 500
global tobs = $nobs + $burn

*** Maximum LP horizon
global hmax 10

*** AR(1) parameter
global rho  0.95

set obs $nreps

*** Generate empty vectors to store responses
forvalues h = 0/$hmax{
	gen r_level_`h' = .
	gen r_ldiff_`h' = .
	gen r_cdiff_`h' = .
	gen r_ar_`h'    = .
}

set seed 12345
qui forvalues i = 1(1)$nreps{
	noi disp "`i'," , _continue
	preserve
	clear
	
	set obs  $tobs
	gen t = _n

	sort t, stable
	tsset t
	*** Generate data from the model
	
	gen vy 	= rnormal()
	gen y 	= 0
	
		*!!! add noise and lagdiff
		gen e=rnormal()
		global a 0.00

	replace y = $rho * l.y + $a * l.d.y + vy + e	if _n > 2
 
	*** Drop the burn-in sample
	drop if _n <= $burn
	replace t = _n 

	sort t, stable
	
	*** Estimate the AR model ***> without knowing the shock
	reg y l.y l.d.y  // cons and Y lag diff
	scalar rhoe = _b[L.y]
	scalar ae = 0
	scalar ae = _b[L.D.y]
	
	local b_ar_0 = 1
	local b_ar_1 = rhoe * `b_ar_0' + ae * (`b_ar_0' - 0)
		
	forvalues h = 2/$hmax{
		local j=`h'-1
		local k=`h'-2
		local b_ar_`h' = rhoe * `b_ar_`j'' + ae * (`b_ar_`j''-`b_ar_`k'')
	}


	*** Estimate the LP in levels
	forvalues h = 0/$hmax{
*		reg f`h'.y y 
*		local b_level_`h' = _b[y] 
		reg f`h'.y vy l.y l.d.y  // cons and Y lag diff
		local b_level_`h' = _b[vy] 

		}
		
	*** Generate long differences
	forvalues h = 0/$hmax{
		gen f`h'y = f`h'.y - l.y
	}	
	
	*** Generate short differences
	forvalues h = 0/$hmax{
		gen df`h'y = f`h'.y - f`h'l.y
	}	
	
	
	
	*** Estimate the LP in long-differences
	forvalues h = 0/$hmax{
*		reg f`h'y d.y 
*		local b_ldiff_`h' = _b[D.y] 
		reg f`h'y vy  l.d.y  // cons and Y lag diff
		local b_ldiff_`h' = _b[vy] 
	}

	
	*** Estimate the LP in cum short-differences
	forvalues h = 0/$hmax{
*		reg f`h'y d.y 
*		local b_ldiff_`h' = _b[D.y] 
		reg df`h'y vy  l.d.y  // cons and Y lag diff
		if `h'==0	local b_cdiff_`h' = _b[vy]
		local j=`h'-1
		if `h'>0	local b_cdiff_`h' = `b_cdiff_`j'' + _b[vy] 
	}	
	
	restore
	forvalues h = 0/$hmax{
		replace r_level_`h' = `b_level_`h'' in `i'
		replace r_ldiff_`h' = `b_ldiff_`h'' in `i'
		replace r_ar_`h'    = `b_ar_`h''    in `i'
		replace r_cdiff_`h' = `b_cdiff_`h'' in `i'

	}
}

gen rl = .
gen rd = .
gen rc = .
gen ra = .

forvalues h = 0/$hmax{
	sum r_level_`h', meanonly
	replace rl = r(mean) if `h' == _n-1
	sum r_ldiff_`h', meanonly
	replace rd = r(mean) if `h' == _n-1
	sum r_cdiff_`h', meanonly
	replace rc = r(mean) if `h' == _n-1
	sum r_ar_`h', meanonly
	replace ra = r(mean) if `h' == _n-1

}

drop if _n > $hmax + 1
gen t = _n-1

sort t, stable
tsset t

*** Generate true response from AR(1)
gen rt = $rho
forvalues i = 0/$hmax{
	replace rt = $rho^`i' if `i' == t
}

*** Generate true response 
cap drop rt
gen rt = 1 if t==0
replace rt = $rho + $a * 1 if t==1
forvalues i = 2/$hmax{
	replace rt = $rho * l.rt + $a * (l.d.rt) if `i' == t
}


label var rt "True"
label var rl "Level"
label var rd "Long-difference"
label var rc "Cumulative-difference"
label var ra "AR(1)"

/*
*** Plot the results from the Monte Carlo
twoway (line rt t, lcolor(black%50) lwidth(medium)) ///
(line rl t, lcolor(blue%50) lpattern(dash)) ///
(line ra t, lcolor(brown%50) lpattern(solid) lwidth(medium)) ///
(line rd t, lpattern(longdash_dot) lcolor(pink%75))  ///
(line rc t, lpattern(shortdash) lcolor(purple%75)  ///
title("AR(1) with parameter $rho", color(black) size(medium)) ///
xtitle("Horizon", size(medsmall)) ///
graphregion(color(white)) plotregion(color(white)) ///
legend(cols(3) region(lstyle(none)) size(medium)) ///
note("Sample size: $nobs        Monte Carlo simulations: $nreps") ///
xlabel(1(1)$hmax) yscale(range(0 1)) ylabel(0(0.2)1))
*/

twoway ///
(line rt t, lcolor(black)       lpattern(solid)        lwidth(thick) ) ///
(line rl t, lcolor(blue%50)     lpattern(dash)         lwidth(medthick) ) ///
(line ra t, lcolor(navy%50)     lpattern(dash_dot)     lwidth(medthick) ) ///
(line rd t, lcolor(pink%75)     lpattern(longdash_dot) lwidth(medthick) )  ///
(line rc t, lcolor(purple%75)   lpattern(shortdash)    lwidth(medthick)    ///
xtit("{it:h}") ///
ytit("{it:{&beta}}{subscript:{it:h}}") ///
graphregion(color(white)) plotregion(color(white)) ///
legend(ring(0) pos(1) cols(2) region(col(none)) order(2 3 4 5 - 1 )) ///
/// legend(off) ///
xsize(3) ysize(3) scale(0.9) ///
xlabel(1(1)$hmax)  ysc(r(0.6 1.2)) ylabel(0.6(0.2)1.0) )

local A $nreps
local B $nobs
local C = $rho*1000

gr export "SSBias_IntcpYLagdiffY_rho_`C'_reps_`A'_obs_`B'.pdf", replace




