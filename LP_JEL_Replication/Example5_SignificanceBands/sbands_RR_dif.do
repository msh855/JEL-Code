*******************************************************
*	SIGNIFICANCE BANDS: Romer and Romer example
*******************************************************
cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example5_SignificanceBands/"

clear
set more off 

set scheme s1color
graph set window fontface "Palatino"

*********DATA************************************

use aggregatedata_final.dta, clear



*** drop any missing observations
cap drop when* 
gen when=0

*** scale variables to work in percent
replace dlrgdp = dlrgdp
replace dlcpi  = dlcpi
replace lcpi   = 100*lcpi
replace lrgdp  = 100*lrgdp


drop if qdate < tq(1985q1)
drop if qdate > tq(2007q4)
sort qdate
tsset qdate, q

*********************************************
**** SET PARAMETERS + RUN SPECIFICATIONS ****
*********************************************

** increase the plot horizon 
local h = 17

* Impulse variable: stir
local x dstir

* Response variables: lcpi
local y dlcpi

* Instrument: rr_shock
local z rr_shock

* LP lags
local lags = 4

* N. of Newey-West lags
local nwlag `h'

* Significance level of test
local p 0.05

* Block size for bootstrap
local block `h'


****** LP Regressions *************************

*** Construct LHS forward variables
forvalues i=0/`h' {

	cap drop `y'_f`i'

	gen `y'_f`i' = f`i'.`y' 
	
	
}

*** Empty variables to fill in with the LP estimates

	cap drop b_`y'_`x' se_`y'_`x'
		
	gen b_`y'_`x' = .
	gen se_`y'_`x' = .
		


qui{
	forvalues i = 0/`h' {
			newey `y'_f`i' `z' ///
			l(1/`lags').dlrgdp l(1/`lags').dlcpi l(1/`lags').dstir, ///
			lag(`nwlag')
			

			replace b_`y'_`x'   = _b[`z']  if _n== `i'+1
			replace se_`y'_`x'  = _se[`z'] if _n== `i'+1

	}
}
*** Generate the usual confidence bands


	cap drop u1_`y'_`x' d1_`y'_`x' u2_`y'_`x' d2_`y'_`x'
	gen u1_`y'_`x' = .
	gen d1_`y'_`x' = .
	gen u2_`y'_`x' = .
	gen d2_`y'_`x' = .
	
	replace u1_`y'_`x' = b_`y'_`x' + se_`y'_`x'
	replace d1_`y'_`x' = b_`y'_`x' - se_`y'_`x'
	replace u2_`y'_`x' = b_`y'_`x' + invnormal(1 -`p'/2)*se_`y'_`x'
	replace d2_`y'_`x' = b_`y'_`x' - invnormal(1 -`p'/2)*se_`y'_`x'
	


****************************************************
*** Now do joint estimation using stacking trick ***
*** Also useful to construct significance bands  ***

*** First do FWL step


qui{
		forvalues i=0/`h'{
		reg `y'_f`i' l(1/`lags').dlrgdp l(1/`lags').dlcpi l(1/`lags').dstir
		predict r_`y'_f`i', residuals
		
		}

	
		reg `z' l(1/`lags').dlrgdp l(1/`lags').dlcpi l(1/`lags').dstir
		predict r_`z', residuals
		

}

*** Note I used Romer shocks directly rather than as an instrument

*** Do Significance bands as in the paper
gen bju = .
gen bjd = .

gen w = r_`z'^2
reg w
local mw = _b[_cons]

forvalues i = 0/`h'{
gen eta_`i' = r_`y'_f`i' * r_`z'
newey eta_`i', lag(`nwlag')
local seta_`i' = _se[_cons]

local sbeta_`i' = `seta_`i''/`mw'
*	Bonferroni adjustment
*	replace bju = invnormal(1 - (`p'/(2*(`h'+1))))*`sbeta_`i'' if `i' == _n -1
*	Usual bands
	replace bju = invnormal(1 - (`p'/2))*`sbeta_`i'' if `i' == _n -1
	replace bjd = -bju
}

*** Do a bootstrap version
gen bootju = .
gen bootjd = .


* Create a variable for the block index
gen block_id = floor((_n+`block'-1) /`block')
gen x = 1

forvalues i = 0/`h'{
wildbootstrap reg eta_`i' x, nocons cluster(block_id)  rseed(12345)
local bootse_`i' = _se[x]
local sboot_b_`i' = `bootse_`i''/`mw'

*	Bonferroni adjustment
*	replace bootju = invnormal(1 - (`p'/(2*(`h'+1))))*`sboot_b_`i'' if `i' == _n -1
*	Usual bands
	replace bootju = invnormal(1 - (`p'/2))*`sboot_b_`i'' if `i' == _n -1
	replace bootjd = -bootju
}



sort qdate
tsset qdate, q


*** stack data by horizon
reshape long r_`y'_f, i(qdate) j(hor)

tsset hor qdate

*** interact intervention with horizon dummy
forvalues i = 0/`h'{
	gen r_`z'_`i' = r_`z'*(hor==`i')
	}

*** variable will store estimates of the response
cap drop bj 
gen bj = .
gen sj = .


*** run the pooled regression
*** Command for OLS with Driscoll-Kraay
xtscc r_`y'_f r_`z'_* , fe lag(`nwlag')

*** Not needed but to check you get the same results as previously
forvalues i = 0/`h'{
	replace bj =  _b[r_`z'_`i'] if _n==`i'+1
	replace sj = _se[r_`z'_`i'] if _n==`i'+1
}



*** Do joint test
testparm r_`z'_*
local pvalue : display %12.3g r(p) 





cap drop t

cap drop zero
gen zero = 0 if _n <= `h'

gen t = _n-1 if _n <= `h'
	 
*** Figure with confidence bands only
		
		twoway (rarea u2_`y'_`x' d2_`y'_`x'  t,  ///
		fcolor(green%10) lcolor(gs15) lw(none) lpattern(solid))  ///
		(rarea u1_`y'_`x' d1_`y'_`x'  t,  ///
		fcolor(green%20) lcolor(gs13) lw(none) lpattern(solid)) ///
		(line b_`y'_`x' t, lcolor(green) ///
		lpattern(solid) lwidth(thick)) /// 
		(line zero t, lcolor(black)), ///
		xlabel(0(5)15) ylab(-6(2)2) ///
		ytitle("Response, log x100") ///
		xtitle("Horizon, quarters, {it:h}") ///
		note("{it:p}-value of joint significance test: `pvalue'") ///
		graphregion(color(white)) plotregion(color(white)) ///
		legend(off)  ///
		xsize(3) ysize(3.75)


		graph rename diff_gc_`y', replace
		//gr save diff_gc_`y'_`h'.gph, replace
		gr export diff_gc_`y'_`h'.pdf, replace
		
*** Figure with significance bands 
		
		twoway ///
		(scatteri 1.8 0 1.8 16,  recast(line) lw(thin) mc(none) lc(black) lp(solid)) ///	
		(scatteri 1.8 0 1.8 16,  recast(dropline) base(1.6) lw(thin) mc(none) lc(black) lp(solid)) ///
		(rarea u2_`y'_`x' d2_`y'_`x'  t,  ///
		fcolor(green%10) lcolor(gs15) lw(none) lpattern(solid))  ///
		(rarea u1_`y'_`x' d1_`y'_`x'  t,  ///
		fcolor(green%20) lcolor(gs13) lw(none) lpattern(solid)) ///
		(line b_`y'_`x' t, lcolor(green) ///
		lpattern(solid) lwidth(thick)) /// 
		(line bju t, lcolor(blue) ///
		lpattern(dash)) ///
		(line bjd t, lcolor(blue) ///
		lpattern(dash)) ///
		(line bootju t, lcolor(red) ///
		lpattern(longdash)) ///
		(line bootjd t, lcolor(red) ///
		lpattern(longdash)) ///
		(line zero t, lcolor(black)), ///
		xlabel(0(5)15) ylab(-6(2)2) ///
		/// subtitle("(b) Inflation response") ///
		ytitle("Response, {&Delta} log CPI x100, {fontface cmsy10: R}({it:h})") ///
		xtitle("Horizon, quarters, {it:h}") ///
		/// note("{it:p}-value of joint significance test: `pvalue'") ///
		text(2.2 8 "{it:p}-value of joint significance test: `pvalue'") ///
		graphregion(color(white)) plotregion(color(white)) ///
		legend(off)  ///
		xsize(3) ysize(3.75)


		graph rename diff_gs_`y', replace
		// gr save diff_gs_`y'_`h'.gph, replace
		gr export diff_gs_`y'_`h'.pdf, replace

		
	/*gr combine diff_gc_`y' gs_`y', ycommon
	gr save diff_gcombi_`y'_`h'.gph, replace
	gr export diff_gcombi_`y'_`h'.pdf, replace
	
	
	gr combine "gs_lcpi_17" diff_gs_`y'_`h'.gph, ycommon
	gr save sig_combi.gph, replace
	gr export sig_combi.pdf, replace
	*/
	