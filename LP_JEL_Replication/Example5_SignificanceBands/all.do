***********************************************
* Example to illustrate significance bands
* with an application using Romer and Romer
***********************************************

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example5_SignificanceBands/"

cap log close
log using all.log , replace

* graph settings
set more off
set scheme s1color

graph set window fontface "Palatino"
graph set window fontfaceserif "Palatino"
 
gr drop _all

* main

do sbands_RR.do
do sbands_RR_dif.do

* clean up and save for use in the paper

copy gs_lcpi_17.pdf  			Figure5a.pdf , replace
copy diff_gs_dlcpi_17.pdf  		Figure5b.pdf , replace

cap log close
