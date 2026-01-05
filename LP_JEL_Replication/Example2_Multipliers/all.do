***********************************************
* Example to show how to directly 
* compute multipliers via LPs
***********************************************
cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/"
cap cd "/Users/amtaylor/Dropbox/JordaTaylorDropbox/JEL_LP/"
cd "LP_JEL_Replication/Example2_Multipliers/"


cap log close
log using all.log , replace

* graph settings
set more off
set scheme s1color

graph set window fontface "Palatino"
graph set window fontfaceserif "Palatino"
 
gr drop _all

* main

do GLP2_responses_Rfull.do
do GLP2_responses_Rboom.do
do GLP2_responses_Rslump.do

do GLP2_responses_Mfull.do
do GLP2_responses_Mboom.do
do GLP2_responses_Mslump.do

* clean up and save for use in the paper

copy Rfull.pdf  Figure2a.pdf , replace
copy Mfull.pdf  Figure2b.pdf , replace

copy Rboom.pdf  Figure9a.pdf , replace
copy Rslump.pdf Figure9b.pdf , replace

cap log close

