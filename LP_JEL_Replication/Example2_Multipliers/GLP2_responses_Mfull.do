cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/LP_JEL_Replication/Example2_Multipliers"

cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/LP_JEL_Replication/Example2_Multipliers/"
clear
use   fiscal_consolidation_v032023.dta

merge 1:1  ifs year   using   JSTdatasetR6.dta , keep(3) nogen

merge 1:1  ifs year   using   dcapb.dta , keep(3) nogen

gen dCAPB = dnlgxqa  

xtset ifs year , yearly

// outcome
gen y = log(rgdpbarro)*100
gen Dy = d.y
// long diffs and sums of long diffs
forv h=0/10{
	gen D`h'y = f`h'.y - l.y
}
forv h=0/10{
	egen S`h'y = rowtotal(D0y-D`h'y)
}

// treatment
forv h=0/10{
	local j = `h'-1
	if `h'==0  gen  dCAPB`h'   = dCAPB
	if `h'>0   gen  dCAPB`h'  = f`h'.dCAPB + dCAPB`j'
}
forv h=0/10{
	egen SdCAPB`h' = rowtotal(dCAPB0-dCAPB`h')
}


// output gap for stratification
gen y_hpcyc=.
gen y_hptrend=.
levelsof iso , local(ctys)
foreach c in `ctys' {
	disp "`c'"
	gen temp0 = y if iso=="`c'"
	gen ifsyear = ifs*10000+year
	tsset ifsyear // temporary
	tsfilter hp temp1 = temp0  , smooth(400) trend(temp2)
	replace y_hpcyc = temp1 if iso=="`c'"
	replace y_hptrend = temp2 if iso=="`c'"
	list iso year y y_hpcyc if temp0~=.
	cap drop temp* ifsyear
}
xtset ifs year , yearly



// controls, 2 lags of outcome and treatment, l.y_hpcyc  ld.debtgdp

gen _x1 = l1d.y
gen _x2 = l2d.y
gen _x3 = l1.dCAPB
gen _x4 = l2.dCAPB
gen _x5 = l.y_hpcyc  
gen _x6 = ld.debtgdp

gen		full = 1
gen		boom  = cond( l.y_hpcyc >  0 , 1, 0 )
gen		slump = cond( l.y_hpcyc <= 0 , 1, 0 )
gen		exp  = cond( ld.y >  1.5 , 1, 0 )
gen		rec  = cond( ld.y <= 1.5 , 1, 0 )

// R full controls
foreach v in h b d u ba da ua Zero {
	//cap drop `v'
	gen      `v' = .
}
eststo clear
forv h=0/4 {
	
	*** LPIV
		xi: xtivreg2 S`h'y (SdCAPB`h' = size) _x*  , fe cluster(iso)
		estimates store LP`h'
		eststo LP`h'
		
	replace h = _n-1						if _n-1==`h'
	replace Zero = 0 						if h==`h' | l.h==`h'
	replace b = _b[SdCAPB] 					if h==`h'
	replace d = b+1.96*_se[SdCAPB]	if h==`h'
	replace u = b-1.96*_se[SdCAPB]	if h==`h'
}
esttab LP* , b(2) se keep(*CAPB*)

//stack
qui{
forv h=0/4 {
preserve	
gen Y = S`h'y
gen T_h`h' = SdCAPB`h'
gen Z_h`h' = size
gen I = ifs
rename _x* X*_h`h'
gen H=`h'
keep Y T* Z* X* I* H boom exp year //*
save stack`h'.dta , replace
restore
}
preserve
clear
forv h=0/4 {
	append using stack`h'.dta
}
forv h=0/4 {
	forv j=0/4 {
		replace T_h`j'=0		if `h'~=`j' & `h'==H
		replace Z_h`j'=0		if `h'~=`j' & `h'==H
	forv k=1/6 {
		replace X`k'_h`j'=0		if `h'~=`j' & `h'==H
	}
	}
}
save stack.dta , replace
gen FE = I*1000+H // e.g. USA at horizon 1 is 111001 etc. 
	*** LPIV stacked
	tsset FE year
	noi xi: ivreg2 Y (T_* = Z_*) X* i.FE , cluster(FE) // dkraay(3) //*
	estimates store stackedLP
	eststo stackedLP
	lincom (T_h0+T_h1+T_h2+T_h3+T_h4)/5
	test T_h0 T_h1 T_h2 T_h3 T_h4
	local df_joint = r(df)
	local chi2_joint = r(chi2)
	local p_joint = r(p)
	
restore
}
esttab stackedLP , b(2) se keep(*T_*)
	lincom (T_h0+T_h1+T_h2+T_h3+T_h4)/5
	replace h=5 if _n-1==5
	replace b=r(estimate) if    h==5
	replace d = b+1.96*r(se)	if h==5
	replace u = b-1.96*r(se)	if h==5

// output summary
esttab LP* , b(2) se keep(*CAPB*)
esttab stackedLP , b(2) se keep(*T_*)
lincom (T_h0+T_h1+T_h2+T_h3+T_h4)/5

// plot b + ci for all h and avg using stacked
estimates restore stackedLP
gen B=.
gen U=.
gen D=.
forv h=0/4 {
	replace B=_b[T_h`h'] 						if h==`h'
	replace U=_b[T_h`h'] + 1.96*_se[T_h`h']		if h==`h'
	replace D=_b[T_h`h'] - 1.96*_se[T_h`h'] 	if h==`h'
}

lincom (T_h0+T_h1+T_h2+T_h3+T_h4)/5
	replace B = r(estimate) if    h==5
	replace D = B+1.96*r(se)	if h==5
	replace U = B-1.96*r(se)	if h==5


format B %5.2f
twoway ///
(scatteri 1.4 0 1.4 4,  recast(line) lw(thin) mc(none) lc(black) lp(solid)) ///
(scatteri 1.4 0 1.4 4,  recast(dropline) base(1.3) lw(thin) mc(none) lc(black) lp(solid)) ///
(rcap  U D  h  if h==5, vert lc(blue%50))  (scatter B h if h==5 , mlab(B) mlabc(blue) mc(blue) ) ///
(rarea U D  h  if h<5,  fcolor(blue%15) lcolor(gs13) lw(none) lpattern(solid)) ///
(line b h if h<5 , lcolor(blue%50) lpattern(solid) lwidth(thick)) ///
(line Zero h , lcolor(black)), legend(off) ///
///title("Responses of log rgdppc to 1% fiscal consolidation", color(black) size(med)) ///
subtitle("", color(black) size(small)) ///
ytitle("Multiplier, {it:m}({it:h})", size(medsmall)) xtitle("Horizon, years, {it:h} ", size(medsmall)) ///
/// note("Notes: 95 percent confidence bands. Joint test of {it:{&beta}{subscript:h}}=0: {it:{&chi}{subscript:`df_joint'}{superscript:2}} =`chi2_joint' ({it:p}=`p_joint').") ///
text(1.9 2 "Joint test, {it:m}({it:h})=0:" " " "{it:{&chi}}{superscript:2}(`df_joint')=`:display %5.1fc `chi2_joint'' ({it:p}=`:display %5.3fc `p_joint'')") ///
xsize(2) ysize(2.3) xsc(r(0 5.4)) xlab(0 1 2 3 4 5 "average") ysc(r(-5 2)) ylab(-5 -4 -3 -2 -1 0 1) scale(1.2) ///
graphregion(color(white)) plotregion(color(white)) ///
name(Mfull,replace)
gr export Mfull.pdf , replace







forv h=0/4 {
 erase stack`h'.dta
}
erase stack.dta

