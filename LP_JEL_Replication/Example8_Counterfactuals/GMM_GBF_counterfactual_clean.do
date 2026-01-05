****************************************************************
* Unemployment rate response to a Romer monetary shock
* Illustration of counterfactual using GBF
* estimation with correct SEs
* Variables involved:
* Unemployment rate
* PCEPI inflation
* Fed Funds Rate

* Romer shock as the instrument
* Use the update from Coibion, Gorodnichenko Kueng and Silvia (JME, 2017) 
****************************************************************
clear

cap cd "/Users/amtaylor/School of Management Dropbox/Alan M. Taylor/JordaTaylorDropbox/JEL_LP/LP_JEL_Replication/Example8_Counterfactuals/"

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/LP_JEL_Replication/Example8_Counterfactuals"


use data_fred.dta, clear

sort date
tsset date, monthly

gen plevel = 100*log(PCEPI) 


* define the response variable
local y urate

* define horizon of impulse response

local hor_u 48
local hor_r `hor_u' 
* define the lags
local lags 6

* define the start/end of the sample
drop if date < tm(1985m1)
drop if date > tm(2000m12)

* Generate forward values for LHS in the LPs
forvalues i= 0/`hor_r' {
	
	cap drop `y'_f`i'
	gen `y'_f`i' = (f`i'.`y' - l.`y') // urate
	gen ffr_f`i' = (f`i'.ffr - l.ffr) // ffr
}

*** Orthogonalize the LHS wrt controls
qui{
	forvalues h = 0/`hor_r'{
		reg `y'_f`h' l(1/`lags').urate l(1/`lags').infl ///
		l(1/`lags').ffr 
		predict r`y'_f`h', resid
		
		reg ffr_f`h' l(1/`lags').urate l(1/`lags').infl ///
		l(1/`lags').ffr
		predict effr_f`h', resid
	}
}

*** Orthogonalize the treatment wrt controls
qui{
		reg ffr l(1/`lags').urate l(1/`lags').infl l(1/`lags').ffr 
		predict rffr, resid
}

*** Orthogonalize the instrument wrt controls
qui{
		reg RRCGShock l(1/`lags').urate l(1/`lags').infl ///
		l(1/`lags').ffr 
		predict rz, resid	
}

preserve

** Estimate LPIV 
** OLD stacking trick 

*** stack data by horizon
reshape long r`y'_f, i(date) j(hor)

tsset hor date

* define the impulse variable
local s rffr
* define the instruments
local z rz 
*** interact intervention and instrument with horizon dummy
forvalues i = 0/`hor_u'{
	gen `s'_`i' = `s'*(hor==`i')
	gen `z'_`i' = `z'*(hor==`i')
	}
*** variable will store estimates of the response
cap drop bj bju bj2u bjd bj2d
gen bj = .
gen bju = .
gen bj2u = .
gen bjd = .
gen bj2d = .
* xtscc r`y'_f `s'_* , fe lag(12)
xtivreg28 r`y'_f (`s'_* = `z'_*), fe gmm bw(12)

forvalues i = 0/`hor_u'{
	replace bj = _b[`s'_`i'] if _n==`i'
	replace bju = _b[`s'_`i'] + _se[`s'_`i'] if _n==`i'
	replace bj2u = _b[`s'_`i'] + 1.96*_se[`s'_`i'] if _n==`i'
	replace bjd = _b[`s'_`i'] - _se[`s'_`i'] if _n==`i'
	replace bj2d = _b[`s'_`i'] - 1.96*_se[`s'_`i'] if _n==`i'

}

/*** Do joint test
testparm rffr_*
local pvalue : display %12.5g r(p) //%12.2e r(p)
*/
gen t = _n
keep bj* t
save ivbet.dta, replace
restore


*** Estimate the GBFs
*** Generate the empty vectors to be filled in
*** Unemployment response
cap drop b_gbf b_gbf_u b_gbf_d
gen b_gbf = .
gen b_gbf_u = .
gen b_gbf_d = .	
gen b_gbf_2u = .
gen b_gbf_2d = .



*** Joint estimation of the two GBFs
* Define system of equations at all horizons
local eqs ""
local k = `hor_r'+1
forv h=1/`k' {
	local j=`h'-1
	local eqs "`eqs' (eq`h': r`y'_f`j' - {a0=1.5}*exp(-1*(({b0=24}-`j')/{c0=10})^2)*rffr) "
}
* disp "`eqs'" //check equations correctly generated

local eqsr ""
forv h=1/`k' {
	local j=`h'-1
	local eqsr "`eqsr' (eqr`h': effr_f`j' - {ar0=2}*exp(-1*(({br0=5}-`j')/{cr0=6})^2)*rffr) "
}
* disp "`eqsr'" //check equations correctly generated

* GMM joint estimation of unemployment and funds rate responses
gmm `eqs' `eqsr' , ///
	instruments(rz) winitial(unadjusted, independent) ///
	wmatrix(unadjusted) ///
	twostep vce(hac nw)
	
* Get variance-covariance matrix	
mat V = get(VCE)
matlist V

* get the sub-blocks to do counterfactual
matselrc V Vrr, row(4,5,6) col(4,5,6)
matselrc V Vuu, row(1,2,3) col(1,2,3)
matselrc V Vru, row(4,5,6) col(1,2,3)
matselrc V Vur, row(1,2,3) col(4,5,6)

* Vectors of coefficients from estimation
mat betu = (_b[/a0]\ _b[/b0]\ _b[/c0])
mat betr = (_b[/ar0]\ _b[/br0]\ _b[/cr0])

* Save estimates for use later
estimates save gmmgbfjoint, replace


* NLCOM
qui{
* Generate unemployment response
forvalues h = 1/`hor_u'{ 
	estimates use gmmgbfjoint
	nlcom (k`h': _b[/a0]*exp(-1*((_b[/b0]-`h')/_b[/c0])^2)), post
	replace  b_gbf   =                    _b[k`h'] if _n == `h'
	replace  b_gbf_u = b_gbf[`h'] + _se[k`h'] if _n == `h'
	replace  b_gbf_d = b_gbf[`h'] - _se[k`h'] if _n == `h'
	replace  b_gbf_2u = b_gbf[`h'] + 1.96*_se[k`h'] if _n == `h'
	replace  b_gbf_2d = b_gbf[`h'] - 1.96*_se[k`h'] if _n == `h'

	}
}

* Counterfactual empty vectors
cap drop bc_gbf br_gbf_c
gen bc_gbf = .
gen br_gbf_c = .

*** Do the counterfactual
estimates use gmmgbfjoint
mat betrc = (_b[/ar0]-0*_se[/ar0]\ _b[/br0]-1*_se[/br0] \ _b[/cr0]+0*_se[/cr0] )
  
** Assess the plausibility of the counterfactual
mat chi2 = (betrc - betr)'*invsym(Vrr)*(betrc - betr)
scalar chival = chi2[1,1]

*di " Plausibility check "

di %5.2f chi2tail(3, chival)

local lucas : di %5.2f chi2tail(3, chival)
// Define the counterfactual
*** Formula based on multivariate normal
mat betuc = betu + Vru'*invsym(Vrr)*(betrc - betr)
*** Formula for conditional variance given counterfactual
mat Vc = Vuu - Vur*invsym(Vrr)*Vru

qui{
forvalues h = 1/`hor_u'{ 
* Generate counterfactual unemployment responses
* Unemployment counterfactual response
	replace bc_gbf = ///
	betuc[1,1]*exp(-1*((betuc[2,1]-`h')/betuc[3,1])^2) ///
	if _n == `h'
* Funds rate counterfactual response
	replace br_gbf_c = ///
	betrc[1,1]*exp(-1*((betrc[2,1]-`h')/betrc[3,1])^2) ///
	if _n == `h'
	}
}

gen t = _n
replace t = t-1
gen zero = 0

save output.dta , replace

****************************************************************/
****************************************************************/
** reload to save time

use  output.dta , clear
merge 1:1 t using ivbet
drop _merge

local horizon `hor_u'
estimates use gmmgbfjoint
estimates replay
drop if t > `horizon'

label var b_gbf "U. Rate"
label var br_gbf "Funds Rate"


local a : di %5.2f  betr[1,1]
local b : di %5.2f  betr[2,1]
local c : di %5.2f  betr[3,1]

local ac : di %5.2f  betrc[1,1]
local bc : di %5.2f  betrc[2,1]
local cc : di %5.2f  betrc[3,1]
local horizon 48
** Replicate GBF Figure from earlier in the text
	twoway ///
	(rarea b_gbf_u b_gbf_d  t, ///
	fcolor(purple%20) lcolor(white) lpattern(solid)) ///
	(rarea b_gbf_2u b_gbf_2d  t, ///
	fcolor(purple%30) lcolor(white) lpattern(solid)) ///
	(line bj t, lcolor(blue) lpattern(dash) lwidth(thick)) ///
	(line b_gbf t, lcolor(purple) lpattern(solid) ///
	lwidth(thick)) /// 
	(line zero t, lcolor(black))   , ///
	ytitle("Response, percentage points, {fontface cmsy10: R}({it:h})" "") ///
	xtitle("Horizon, months, {it:h}") ///
		///subtitle("(a) Raw LP vs. GBF approximation") ///
	graphregion(color(white)) plotregion(color(white)) ///
	xlab(0(8)`horizon') ///
	legend(order (4 "GBF" 3 "LP") ///
	cols(3) region(lstyle(none)) ///
	size(small) pos(6))  ///
	name(urateffr, replace)	xsize(3) ysize(3)
	gr rename g1, replace
	
	gr export "counterfig_a.pdf", replace
	gr export "Figure8a.pdf", replace

	
***************** Counterfactual figure ************************
	twoway ///
	(rarea b_gbf_u b_gbf_d  t, ///
	fcolor(purple%20) lcolor(white) lpattern(solid)) ///
	(rarea b_gbf_2u b_gbf_2d  t, ///
	fcolor(purple%30) lcolor(white) lpattern(solid)) ///
	(line b_gbf t, lcolor(purple) lpattern(solid) ///
	lwidth(thick)) ///  
	(line bc_gbf t, lcolor(green) lpattern(dash) lwidth(thick)) /// 
	(line zero t, lcolor(black))   , ///
	ytitle("Response, percentage points, {fontface cmsy10: R}({it:h})" "") ///
	xtitle("Horizon, months, {it:h}") ///
	///subtitle("(b) GBF response vs. counterfactual") ///
	graphregion(color(white)) plotregion(color(white)) ///
	xlab(0(8)`horizon') ///
	legend(order (3 "GBF" 4 "counterfactual") cols(2) region(lstyle(none)) ///
	size(small) pos(6)) ///
	name(urateffrc, replace)	xsize(3) ysize(3)
	gr rename g2, replace
	
	gr export "counterfig_b.pdf", replace
	gr export "Figure8b.pdf", replace

	gr combine g1 g2, ycommon ///
	note("Plausibility test p-value: `lucas'") ///
	graphregion(color(white)) plotregion(color(white))
	
	
	graph save   counterfig.gph, replace
	gr export "counterfig.pdf", replace

