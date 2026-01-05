
cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example7_MinimumDistance/"


import excel monthlyShocks.xlsx , first clear
keep Date Assignedtomonths
drop if Date==.
gen month = mofd(Date)
drop Date
format month %tm
order month
rename Assignedtomonths shock_m
save monthlyShocks.dta , replace

import excel quarterlyShocks.xlsx , first clear
keep Date Assignedtoquarters
drop if Date==.
gen quarter = qofd(Date)
drop Date
format quarter %tq
order quarter
rename Assignedtoquarters shock_q
save quarterlyShocks.dta , replace

import excel regressionData.xlsx , first clear sheet("Monthly UK")
drop if Date==.
gen month = mofd(Date)
drop Date
format month %tm
order month
save monthlyData.dta , replace

import excel regressionData.xlsx , first clear sheet("Quarterly UK")
drop if Date==.
gen quarter = (Date-1975)*4 + tq(1975q1)
drop Date
format quarter %tq
order quarter
save quarterlyData.dta , replace


* oil price and NEER as additional controls

clear
freduse WTISPLC CCUSSP01GBM650N GBRGDPDEFQISMEI NNGBBIS

gen logNarrowNEER = log(NNGBBIS)
gen logoilpriceusd = log(WTISPLC)
gen logoilpricegbp = log(WTISPLC/CCUSSP01GBM650N)
gen loggdpdeflator = log(GBRGDPDEFQISMEI)
gen month = mofd(daten)
format month %tm
tsset month
ipolate loggdpdeflator month , gen (loggdpdeflator_m)

gen logrpoiluk = logoilpricegbp - loggdpdeflator_m

center logrpoiluk logNarrowNEER , inplace

keep month logrpoil logNarrowNEER
tsline logrpoiluk logNarrowNEER

save logOilNEER.dta , replace
