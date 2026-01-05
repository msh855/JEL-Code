
cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example6_JointInference/"


****************************************************************/
****************************************************************/
** reload to save time

use  output.dta , clear
local horizon 48
estimates use gmm
estimates replay
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

*local pvalue : display %12.5f r(p) //%12.2e r(p)
local p_joint: di %5.3f r(p)
disp "`p_joint'"
local chi2_joint: di %5.3f r(chi2)
disp "`chi2_joint'"
local df_joint: di r(df)
disp "`df_joint'"


estimates use gmmgbf
estimates replay

local ab:  di %5.3fc  _b[a0:_cons]
local ase: di %5.3fc _se[a0:_cons]
local bb:  di %5.3fc  _b[b0:_cons]
local bse: di %5.3fc _se[b0:_cons]
local cb:  di %5.3fc  _b[c0:_cons]
local cse: di %5.3fc _se[c0:_cons]

disp "a = `ab' (`ase')" 
disp "b = `bb' (`bse')" 
disp "c = `cb' (`cse')" 



***************** LPIV figure ********************************
	twoway ///
	(scatteri 2.3 0 2.3 48,  recast(line) lw(thin) mc(none) lc(black) lp(solid)) ///
	(scatteri 2.3 0 2.3 48,  recast(dropline) base(2.2) lw(thin) mc(none) lc(black) lp(solid)) ///
	(rarea bju bjd  t, fcolor(blue%10) lcolor(white) lpattern(solid)) ///
	(rarea bj2u bj2d  t, fcolor(blue%20) lcolor(white) lpattern(solid)) ///
	(line bj t, lcolor(blue) lpattern(dash) lwidth(thick)) /// 
	(line zero t, lcolor(black)) if t < `horizon'  , ///
	/// title("Response estimated by LPIV", color(black) size(medium)) ///
	ytitle("Response, percentage points, {fontface cmsy10: R}({it:h})") xtitle("Horizon, months, {it:h}") ///
	graphregion(color(white)) plotregion(color(white)) ///
		text(2.5 24 "Joint test, {fontface cmsy10: R}({it:h})=0:  {it:{&chi}}{superscript:2}(`df_joint')=`:display %5.1fc `chi2_joint'' ({it:p}=`:display %5.3fc `p_joint'')") ///
		xlab(0(12)48) ylab(-1(1)2.5)	legend(off) name(fig_urate_LPIV, replace) ///
		xsize(3) ysize(3)

	//graph save   fig_urate_LPIV.gph, replace
	gr export "fig_urate_LPIV.pdf", replace

***************** GBF with GBR bands **************************	
 	twoway (rarea b_gbf_u b_gbf_d  t, ///
	fcolor(purple%10) lcolor(white) lpattern(solid)) ///
	(rarea b_gbf_2u b_gbf_2d t, ///
	fcolor(purple%20) lcolor(white) lpattern(solid)) ///
	(line bj t, lcolor(blue) ///
	lpattern(dash) lwidth(thick)) /// 
	(line b_gbf t, lcolor(purple) ///
	lpattern(solid) lwidth(thick)) /// 
	(line zero t, lcolor(black)) if t < `horizon', ///
	/// title("Gaussian Basis Function Approximation", color(black) size(medium)) ///
	ytitle("Response, percentage points, {fontface cmsy10: R}({it:h})") xtitle("Horizon, months, {it:h}") ///
	graphregion(color(white)) plotregion(color(white)) ///
	text( 2.6 32 	"GBF parameters:" , placement(east) ) ///
	text( 2.42 32 	"{it:a}=`ab' (`ase')" , placement(east) ) ///
	text( 2.24 32 	"{it:b}=`bb' (`bse')" , placement(east) ) ///
	text( 2.06 32 	"{it:c}=`cb' (`cse')" , placement(east) ) ///
	xlab(0(12)48) ylab(-1(1)2.5)	legend(off) name(fig_urate_gbf, replace) ///
		xsize(3) ysize(3)
		
	//graph save   fig_urate_gbf.gph, replace
	gr export "fig_urate_gbf.pdf", replace
	

	*************************************************************


	