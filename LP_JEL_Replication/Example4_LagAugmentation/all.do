***********************************************
* Example to show coverage of error bands
* Newey-West vs lag-augmentation
***********************************************
cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example4_LagAugmentation/"

cap log close
log using all.log , replace

* graph settings
set more off
set scheme s1color

graph set window fontface "Palatino"
graph set window fontfaceserif "Palatino"
 
gr drop _all

* main

do nw_v_la.do

* clean up and save for use in the paper

copy fig_nw_v_la.pdf  Figure4.pdf , replace

cap log close

