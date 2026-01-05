***********************************************
* Example of a GBF approximation
***********************************************

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example3_Smoothing/"

cap log close
log using all.log , replace

* graph settings
set more off
set scheme s1color

graph set window fontface "Palatino"
graph set window fontfaceserif "Palatino"
 
gr drop _all

* main

do GBF

* clean up and save for use in the paper

copy  GBF_plot.pdf       Figure3.pdf , replace

cap log close

