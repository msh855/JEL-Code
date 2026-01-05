****************************************
*** Small sample bias illustration
****************************************

* 100 reps takes 100 seconds approx (M2 MacMini)

cap cd "/Users/amtaylor/School of Management Dropbox/Alan M. Taylor/JordaTaylorDropbox/JEL_LP/LP_JEL_Replication/Example1_LongDifferences/"

cap cd "/Users/oscar/Dropbox/JordaTaylorDropbox (1)/JEL_LP/LP_JEL_Replication/Example1_LongDifferences"

cap log close
log using all-simulate.log , replace

di " `c(current_date)' `c(current_time)'"


// small sample
global nobs 100
global nreps 10000 

do SSBias_IntcpNLagdiffN_95
save SSBias_IntcpNLagdiffN_95.dta , replace 

do SSBias_IntcpNLagdiffY_95
save SSBias_IntcpNLagdiffY_95.dta , replace 

do SSBias_IntcpYLagdiffN_95
save SSBias_IntcpYLagdiffN_95.dta , replace 

do SSBias_IntcpYLagdiffY_95
save SSBias_IntcpYLagdiffY_95.dta , replace 


do SSBias_IntcpNLagdiffN_100
save SSBias_IntcpNLagdiffN_100.dta , replace 

do SSBias_IntcpNLagdiffY_100
save SSBias_IntcpNLagdiffY_100.dta , replace 

do SSBias_IntcpYLagdiffN_100
save SSBias_IntcpYLagdiffN_100.dta , replace 

do SSBias_IntcpYLagdiffY_100
save SSBias_IntcpYLagdiffY_100.dta , replace 


/*
// large sample
global nobs 100000
global nreps 100

do SSBias_IntcpNLagdiffN_95
do SSBias_IntcpNLagdiffY_95
do SSBias_IntcpYLagdiffN_95
do SSBias_IntcpYLagdiffY_95

do SSBias_IntcpNLagdiffN_100
do SSBias_IntcpNLagdiffY_100
do SSBias_IntcpYLagdiffN_100
do SSBias_IntcpYLagdiffY_100
*/

di " `c(current_date)' `c(current_time)'"

cap log close

