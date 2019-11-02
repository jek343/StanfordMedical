param money;
param increment;
param limit;

set Intervals := increment .. limit by increment;

set States;

set Counties;

param Pairs {Counties, States};

param entry_cost {s in States};

param Char_1 {c in Counties};

param Char_2 {c in Counties};
param Char_3 {c in Counties};

param benefit {c in Counties, i in Intervals} = Char_3[c] * 1/(1 + Char_1[c]*(1.001)^(Char_2[c]*-1*i));
#Char_1[c] * Char_2[c] * sqrt(Char_3[c]) * log(i) / exp(Char_3[c]) * log10(Char_1[c])

var amt_per_county {c in Counties, i in Intervals} binary;
var entry_state {s in States} binary;

maximize Total_Aid: sum{c in Counties, i in Intervals} amt_per_county[c,i] * benefit[c,i];

#Only spend one amount at that county
subject to One_Spend {c in Counties}: sum{i in Intervals} amt_per_county[c, i] <= 1;

#constraint on the amount of money that you can spend
subject to Money_Cap: sum{c in Counties, i in Intervals} amt_per_county[c, i] * i + sum{s in States} entry_state[s] * entry_cost[s] <= money; 

#prevents spending in state without spending entry cost
subject to Entry_fee {s in States}: entry_state[s] * (sum{c in Counties, i in Intervals} amt_per_county[c,i] * Pairs[c, s]) >= 
																	sum{c in Counties, i in Intervals} amt_per_county[c,i] * Pairs[c, s];