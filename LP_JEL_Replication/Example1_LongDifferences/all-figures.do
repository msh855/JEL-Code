****************************************
*** Small sample bias illustration
****************************************

* 100 reps takes 100 seconds approx (M2 MacMini)

cap cd "/Users/amtaylor/School of Management Dropbox/Alan M. Taylor/JordaTaylorDropbox/JEL_LP/LP_JEL_Replication/Example1_LongDifferences/"

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/LP_JEL_Replication/Example1_LongDifferences"

cap log close
log using all-figures.log , replace

di " `c(current_date)' `c(current_time)'"

use SSBias_IntcpYLagdiffN_95.dta , clear 

label var rt "True"
label var rl "Level"
label var rd "Long-difference"
label var ra "AR(1)"

twoway ///
(line rt t, lcolor(green%30)       	lpattern(solid)		lwidth(thick) ) ///
(line rl t, lcolor(blue%50)     	lpattern(longdash)	lwidth(medthick) ) ///
(line ra t, lcolor(red%50)     		lpattern(dash_dot)		lwidth(medthick) ) ///
(line rd t, lcolor(purple%80)     	lpattern(shortdash)	lwidth(medthick)     ///
xtit("Horizon, {it:h}") ///
ytit("Response, {it:{&beta}}{subscript:{it:h}}" " ") ///
graphregion(color(white)) plotregion(color(white)) ///
legend(ring(0) pos(1) cols(3) region(col(none)) ///
	order(- "Large Sample:" - 1   - "Small sample:" - 2 - - 4 - - 3   )) ///
/// legend(off) ///
xsize(3) ysize(3) scale(1.1) ///
xlabel(0(1)10)  ysc(r(0.6 1.2)) ylabel(0.6(0.2)1.0) name(a, replace) )

gr export SSBiasFigure_a.pdf , replace


use SSBias_IntcpYLagdiffN_100.dta , clear 

label var rt "True"
label var rl "Level"
label var rd "Long-difference"
label var ra "AR(1)"

twoway ///
(line rt t, lcolor(green%30)       	lpattern(solid)		lwidth(thick) ) ///
(line rl t, lcolor(blue%50)     	lpattern(longdash)	lwidth(medthick) ) ///
(line ra t, lcolor(red%50)     		lpattern(dash_dot)		lwidth(medthick) ) ///
(line rd t, lcolor(purple%80)     	lpattern(shortdash)	lwidth(medthick)     ///
xtit("Horizon, {it:h}") ///
ytit("Response, {it:{&beta}}{subscript:{it:h}}" " ") ///
graphregion(color(white)) plotregion(color(white)) ///
legend(ring(0) pos(1) cols(3) region(col(none)) ///
	order(- "Large Sample:" - 1   - "Small sample:" - 2 - - 4 - - 3   )) ///
/// legend(off) ///
xsize(3) ysize(3) scale(1.1) ///
xlabel(0(1)10)  ysc(r(0.6 1.2)) ylabel(0.6(0.2)1.0) name(b, replace) )

gr export SSBiasFigure_b.pdf , replace

* clean up and save for use in the paper

copy  SSBiasFigure_a.pdf       Figure1a.pdf , replace
copy  SSBiasFigure_b.pdf       Figure1b.pdf , replace


di " `c(current_date)' `c(current_time)'"

cap log close

