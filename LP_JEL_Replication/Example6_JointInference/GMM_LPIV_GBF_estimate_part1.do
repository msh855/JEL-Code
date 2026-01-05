****************************************************************
* Unemployment rate response to a Romer monetary shock
* Illustration of joint LPIV estimation and direct GBF
* estimation with correct SEs
* Variables involved:
* Unemployment rate
* PCEPI inflation
* Fed Funds Rate
* Romer shock as the instrument
* Using the update from Coibion, Gorodnichenko Kueng and Silvia (JME, 2017) 
****************************************************************

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example6_JointInference/"

use data_fred.dta, clear

sort date
tsset date, monthly



* define the response variable
local y urate

* define horizon of impulse response
local horizon 48

* define the lags
local lags 6

* define the start/end of the sample
drop if date < tm(1985m1)
drop if date > tm(2000m1)

* Generate forward values for LHS in the LPs
forvalues i= 0/`horizon' {
	
	cap drop `y'_f`i'
	
	gen `y'_f`i' = (f`i'.`y' - l.`y') 
	
}




*** Orthogonalize the LHS wrt controls
qui{
	forvalues h = 0/`horizon'{
		reg `y'_f`h' l(1/`lags').urate l(1/`lags').infl l(1/`lags').ffr 
		predict r`y'_f`h', resid
	}
}
*** Orthogonalize the treatment wrt controls
qui{
		reg ffr l(1/`lags').urate l(1/`lags').infl l(1/`lags').ffr 
		predict rffr, resid
	
}

*** Orthogonalize the instrument wrt controls
qui{
		reg RRCGShock l(1/`lags').urate l(1/`lags').infl l(1/`lags').ffr 
		predict rz, resid
	
}

cap drop bj bju bj2u bjd bj2d
gen bj = .
gen bju = .
gen bj2u = .
gen bjd = .
gen bj2d = .

* Define system of equations at all horizons
local eqsiv ""
forv h=1/49 {
	local j=`h'-1
	local eqsiv "`eqsiv' (eq`h': r`y'_f`j' - {b`j'}*rffr -{c`j'}) "
}
disp "`eqsiv'"

* GMM
gmm `eqsiv' , ///
	instruments(rz l(1/`lags').rz) winitial(unadjusted, independent) ///
	wmatrix(unadjusted) twostep vce(hac nw `lags')

	forvalues i=0/48{
	replace bj = _b[/b`i'] if _n == `i'
	replace bju = _b[/b`i'] + _se[/b`i'] if _n == `i'
	replace bjd = _b[/b`i'] - _se[/b`i'] if _n == `i'
	replace bj2u = _b[/b`i'] + 1.96*_se[/b`i'] if _n == `i'
	replace bj2d = _b[/b`i'] - 1.96*_se[/b`i'] if _n == `i'
	}
test ///
_b[/b0] = _b[/b1] = _b[/b2] = _b[/b3] = _b[/b4] = _b[/b5] = ///
_b[/b6] = _b[/b7] = _b[/b8] = _b[/b9] = _b[/b10] = _b[/b11] = ///
_b[/b12] = _b[/b13] = _b[/b14] = _b[/b15] = _b[/b16] = _b[/b17] = ///
_b[/b18] = _b[/b19] = _b[/b20] = _b[/b21] = _b[/b22] = _b[/b23] = ///
_b[/b24] = _b[/b25] = _b[/b26] = _b[/b27] = _b[/b28] = _b[/b29] = ///
_b[/b30] = _b[/b31] = _b[/b32] = _b[/b33] = _b[/b34] = _b[/b35] = ///
_b[/b36] = _b[/b37] = _b[/b38] = _b[/b39] = _b[/b40] = _b[/b41] = ///
_b[/b42] = _b[/b43] = _b[/b44] = _b[/b45] = _b[/b46] = _b[/b47] = ///
_b[/b48] = 0

estimates save gmm , replace

local pvalue : display %12.5g r(p) //%12.2e r(p)

***** Use GMM to estimate GBF in one go ****************


cap drop b_gbf b_gbf_u b_gbf_d
gen b_gbf = .
gen b_gbf_u = .
gen b_gbf_d = .	
gen b_gbf_2u = .
gen b_gbf_2d = .



* Define system of equations at all horizons
local eqs ""
forv h=1/49 {
	local j=`h'-1
	local eqs "`eqs' (eq`h': r`y'_f`j' - {a0=1}*exp(-1*(({b0=24}-`j')/{c0=10})^2)*rffr) "
}
disp "`eqs'"

* GMM
gmm `eqs' , ///
	instruments(rz) winitial(unadjusted, independent) wmatrix(unadjusted) ///
	twostep vce(hac nw)

estimates save gmmgbf , replace


* NLCOM
qui{
forvalues h = 1/`horizon'{ 
	estimates use gmmgbf
	nlcom (k`h': _b[/a0]*exp(-1*((_b[/b0]-`h')/_b[/c0])^2)), post
	replace  b_gbf   =                    _b[k`h'] if _n == `h'
	replace  b_gbf_u = b_gbf[`h'] + _se[k`h'] if _n == `h'
	replace  b_gbf_d = b_gbf[`h'] - _se[k`h'] if _n == `h'
	replace  b_gbf_2u = b_gbf[`h'] + 1.96*_se[k`h'] if _n == `h'
	replace  b_gbf_2d = b_gbf[`h'] - 1.96*_se[k`h'] if _n == `h'

	}
}

/************ OLD stacking trick *********************
*** stack data by horizon
reshape long r`y'_f, i(date) j(hor)

tsset hor date

* define the impulse variable
local s rffr
* define the instruments
local z rz 
*KuttnerShock is direct from Kuttner's website based on his (2001 JME) paper
*RomerShock from D. Romer's website from AER article




*** interact intervention and instrument with horizon dummy
forvalues i = 0/`horizon'{
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

*** run the pooled regression
*** Command for OLS with Driscoll-Kraay
* xtscc rurate_f `s'_* , fe lag(12)

*** LPIV jointly **************

xtivreg28 rurate_f (`s'_* = `z'_*), fe gmm bw(12)

forvalues i = 0/`horizon'{
	replace bj = _b[`s'_`i'] if _n==`i'
	replace bju = _b[`s'_`i'] + _se[`s'_`i'] if _n==`i'
	replace bj2u = _b[`s'_`i'] + 1.96*_se[`s'_`i'] if _n==`i'
	replace bjd = _b[`s'_`i'] - _se[`s'_`i'] if _n==`i'
	replace bj2d = _b[`s'_`i'] - 1.96*_se[`s'_`i'] if _n==`i'

}

*** Do joint test
testparm rffr_*
local pvalue : display %12.5g r(p) //%12.2e r(p)

****************************************************************/

gen t = _n
replace t = t-1
gen zero = 0

save output.dta , replace

