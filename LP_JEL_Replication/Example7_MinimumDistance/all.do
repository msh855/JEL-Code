***********************************************
* Example to illustrate minimum distance
* with an application to UK Phillips curve
***********************************************

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example7_MinimumDistance/"

cap log close
log using all.log , replace

* graph settings
set more off
set scheme s1color

graph set window fontface "Palatino"
graph set window fontfaceserif "Palatino"
 
gr drop _all

* main

//do make_data          // uncomment to re-make the data
do UK_Phillips_Curve

* clean up and save for use in the paper

copy  Rinfl.pdf       Figure7a.pdf , replace
copy  Rugap.pdf	      Figure7b.pdf  , replace
copy  inflexpinfl.pdf Figure7c.pdf , replace
copy  influgap.pdf    Figure7d.pdf , replace

cap log close


