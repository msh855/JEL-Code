***********************************************
* Example to illustrate joint inference
* with an application using Romer and Romer
***********************************************

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example6_JointInference/"

cap log close
log using all.log , replace

* graph settings
set more off
set scheme s1color

graph set window fontface "Palatino"
graph set window fontfaceserif "Palatino"
 
gr drop _all

* main

// uncomment if needed, but this is slow:
//do GMM_LPIV_GBF_estimate_part1

do GMM_LPIV_GBF_output_part2

* clean up and save for use in the paper

copy fig_urate_LPIV.pdf  		Figure6a.pdf , replace
copy fig_urate_gbf.pdf  		Figure6b.pdf , replace

cap log close


