
***********************************************
* Example to illustrate minimum distance
* with an application to UK Phillips curve
***********************************************

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example7_MinimumDistance/"


clear
use monthlyData.dta, clear
merge 1:1 month using monthlyShocks.dta , nogen
merge 1:1 month using logOilNEER.dta , nogen

tsset month , monthly

list
sum

gen urate = UnempRate
gen policyrate=BankRate

// u* from very lowpass filter
tsfilter bk ustar_cyc_BK = urate , trend(ustar) max(480) 
gen ugap = urate - ustar

gen infla = 100*(ln(CPIindex) - ln(l12.CPIindex))
//gen infla = 100*(ln(RPIX) - ln(l12.RPIX)) // use this line for RPI measure

* use RE and replace infl expected with next period infl actual
gen infle = f1.infla

gen rpoil = 100*logrpoiluk
gen drpoil = d.rpoil
gen dlNEER = d.logNarrowNEER

// potential extra controls for UK open economy factors
gen X1 = drpoil
gen X2 = dlNEER

// CH shock
gen Shock=shock_m 

// plot data
tsline urate ustar  infla Shock if Shock~=. , name(data, replace)  
gr export data.pdf , replace

**************************************************************

gen date=month
format date %tm

sort date
tsset date


*** LPs for infla infle and ugap

* define horizon of impulse response
local horizon 17 

* define the lags
local lags 4


foreach y of varlist infla infle ugap {
	forvalues i = 0/`horizon'{
	gen `y'_f`i' = f`i'.`y'	
	}
}


*** Orthogonalize the LHS wrt controls
qui{
	foreach y of varlist infla infle ugap {
		forvalues h = 0/`horizon'{
			//reg `y'_f`h' l(1/`lags').infla l(1/`lags').infle l(1/`lags').ugap 
			reg `y'_f`h' l(1/`lags').infla l(1/`lags').infle l(1/`lags').ugap l(1/`lags').X1  l(1/`lags').X2
			predict r`y'_f`h', resid
		}
	}
}
*** Orthogonalize the treatment wrt controls
qui{
		//reg policyrate l(1/`lags').infla l(1/`lags').infle l(1/`lags').ugap 
		reg policyrate l(1/`lags').infla l(1/`lags').infle l(1/`lags').ugap l(1/`lags').X1  l(1/`lags').X2
		predict rpolicyrate, resid
	
}



*** Orthogonalize the instrument wrt controls
qui{
		//reg Shock l(1/`lags').infla l(1/`lags').infle l(1/`lags').ugap 
		reg Shock l(1/`lags').infla l(1/`lags').infle l(1/`lags').ugap l(1/`lags').X1  l(1/`lags').X2
		predict rz, resid
	
}


cap drop b_infla b_infle b_ugap
gen b_infla = .
gen b_infle = .
gen b_ugap  = .


* prep for equations

local eqsiv1 ""
local h1 = `horizon'+1
forv h=1/`h1' {
	local j=`h'-1
	local eqsiv1 "`eqsiv1' (eq`h': rinfla_f`j' - {binfla`j'}*rpolicyrate -{cinfla`j'}) "
}

local eqsiv2 ""
local h11 = `h1'+1
local h2  = 2*`horizon' + 2
forv h=`h11'/`h2' {
	local j=`h'-`h11'
	local eqsiv2 "`eqsiv2' (eq`h': rinfle_f`j' - {binfle`j'}*rpolicyrate -{cinfle`j'}) "
}

local eqsiv3 ""
local h21 = `h2'+1
local h3  = 3*`horizon' + 3
forv h=`h21'/`h3' {
	local j=`h'-`h21'
	local eqsiv3 "`eqsiv3' (eq`h': rugap_f`j' - {bugap`j'}*rpolicyrate -{cugap`j'}) "
}

* Use to verify equations correctly specified
*disp "`eqsiv1'"
*disp "`eqsiv2'"
*disp "`eqsiv3'"

* prep for row and column drops
local rcdrop ""
local first = 1
local last  = 2*`horizon'+1
forv h=`first'(2)`last' {
	if `h'>`first'  local rcdrop "`rcdrop',"
	local rcdrop "`rcdrop'`h'"
}
disp "`rcdrop'"


* GMM estimates of each LP

gmm `eqsiv1' , ///
	instruments(rz l(1/`lags').rz) winitial(unadjusted, independent) ///
	wmatrix(unadjusted) twostep vce(hac nw `lags')
	*onestep vce(hac nw `lags')
mat V = get(VCE)
matselrc V V1, row(`rcdrop') col(`rcdrop')
mat list V1
mat b = e(b)
matselrc b b1, col(`rcdrop')
mat list V1
mat list b1

gmm `eqsiv2' , ///
	instruments(rz l(1/`lags').rz) winitial(unadjusted, independent) ///
	wmatrix(unadjusted) twostep vce(hac nw `lags')
	*onestep vce(hac nw `lags')
mat b = e(b)
matselrc b b2, col(`rcdrop')
mat list b2

gmm `eqsiv3' , ///
	instruments(rz l(1/`lags').rz) winitial(unadjusted, independent) ///
	wmatrix(unadjusted) twostep vce(hac nw `lags')
	*onestep vce(hac nw `lags')
mat b = e(b)
matselrc b b3, col(`rcdrop')
mat list b3

* MD regression and point estimates

mat B = b1',b2',b3'
cap drop B*
svmat B
reg B1 B2 B3
mat b = e(b)
matselrc b theta, col(1,2)
mat list theta


* make matrices for VCV

mat G = b1',b2'
mat Omega_R = V1
mat W = invsym(Omega_R)

mat list G
mat list Omega_R
mat list W

mat Omega_theta = invsym(G'*W*G) * (G'*W*Omega_R*W*G) * invsym(G'*W*G)
mat list Omega_theta


mat S = cholesky(Omega_theta)   
mat se_theta = vecdiag(S)  

mat list theta
mat list se_theta


* partial scatters for minimum distance
local horizon 17 

label var B1 "π"
label var B2 "πe"
label var B3 "u-u*"
reg B1 B2 B3
cap drop h
local H = `horizon'+1
gen h=_n-1 if _n<=`H'

local theta1 : display %4.3f  theta[1,1]
local theta2 : display %4.3f  theta[1,2]
local setheta1 : display %4.3f  se_theta[1,1]
local setheta2 : display %4.3f  se_theta[1,2]

avplot B2 , name(inflexpinfl, replace) ///
	note("") ///
	xtit("{it: E( π{superscript:e} | X } )") ytit("{it: E( π | X ) }") mlab(h) ///
	xsize(3) ysize(3) graphregion(color(white)) plotregion(color(white)) ///
	ylab(-.3 (.1) .4) ///
	text(.4 -.4 "Coefficient: `theta1'" "std. error: (`setheta1')" , place(e) justification(left)) 
	gr export inflexpinfl.pdf , replace

avplot B3 , name(influgap, replace) ///
	note("") ///
	xtit("{it: E( x | X ) }") ytit("{it: E( π | X ) }") mlab(h) ///
	xsize(3) ysize(3) graphregion(color(white)) plotregion(color(white)) ///
	ylab(-.3 (.1) .2) ///
	text(.2 .02 "Coefficient: `theta2'" "std. error: (`setheta2')" , place(e) justification(left)) 
	gr export influgap.pdf , replace

line B1 h , lc(blue) lw(medthick) ///
	name(Rinfl, replace) ///
	xsize(3) ysize(3) graphregion(color(white)) plotregion(color(white)) ///
	ytit("Response, inflation {it: π}, log x100, {fontface cmsy10: R}({it:h})") xtit("Horizon , {it:h}") ///
	xlab(0(1)`H') yline(0,lc(black))
	gr export Rinfl.pdf , replace

line B2 h   , lc(blue) lw(medthick) ///
	name(Rexpinfl, replace) ///
	xsize(3) ysize(3) graphregion(color(white)) plotregion(color(white)) ///
	ytit("Response, expected inflation {it: π{superscript:e}}, log x100, {fontface cmsy10: R}({it:h})") xtit("Horizon, months, {it:h}") ///
	xlab(0(1)`H') yline(0,lc(black))
	gr export Rexpinfl.pdf , replace

line B3 h   , lc(blue) lw(medthick) ///
	name(Rugap, replace) ///
	xsize(3) ysize(3) graphregion(color(white)) plotregion(color(white)) ///
	ytit("Response, unemployment gap {it: x}, percent, {fontface cmsy10: R}({it:h})") xtit("Horizon, months, {it:h}") ///
	xlab(0(1)`H') yline(0,lc(black))
	gr export Rugap.pdf , replace


	