***********************************************
* Example of a GBF approximation
***********************************************

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example3_Smoothing/"

clear
set obs  31

local a = 1
local b = 8
local c = 8
local d = `a'/2
gen horizon = _n-1

gen Response = `a'*exp(-(horizon - `b')^2/(`c'^2))

line Response horizon, /// 
	xsize(6) ysize(3) ///
	ytit("Response, {it:{&phi}}({it:h}; {it:{bf:{&Theta}}})") xtit("Horizon, {it:h}") ///
	graphregion(color(white)) plotregion(color(white)) ///
	lc(blue) lwidth(thick) ylabel(,nogrid) ///
	ylab(0(.5)1.3) ///
	text(1.2 20 "GBF parameters:" "{it:a} = `a'" "{it:b} = `b'" "{it:c} = `c'" , place(e) just(left)) ///
	/// note("a = `a'; b = `b'; c = `c'") ///
	xline(`b', lpattern(shortdash) lcolor(gray) lwidth(medthin)) ///
	yline(`d', lpattern(shortdash) lcolor(gray) lwidth(medthin)) ///
	yline(`a', lpattern(shortdash) lcolor(gray) lwidth(medthin))

gr export "GBF_plot.pdf", replace
