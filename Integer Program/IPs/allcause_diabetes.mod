param money;
param increment;
param limit;

set Intervals := increment .. limit by increment;

set States;

set Counties;

param Pairs {Counties, States};

param entry_cost {s in States};

param Poororfairhealthrawvalue {c in Counties};
param Poorphysicalhealthdaysrawvalue {c in Counties};
param Poormentalhealthdaysrawvalue {c in Counties};
param Lowbirthweightrawvalue {c in Counties};
param Adultsmokingrawvalue {c in Counties};
param Adultobesityrawvalue {c in Counties};
param Foodenvironmentindexrawvalue {c in Counties};
param Physicalinactivityrawvalue {c in Counties};
param Accesstoexerciseopportunitiesrawvalue {c in Counties};
param Excessivedrinkingrawvalue {c in Counties};
param Sexuallytransmittedinfectionsrawvalue {c in Counties};
param Teenbirthsrawvalue {c in Counties};
param Uninsuredrawvalue {c in Counties};
param Primarycarephysiciansrawvalue {c in Counties};
param Dentistsrawvalue {c in Counties};
param Preventablehospitalstaysrawvalue {c in Counties};
param Mammographyscreeningrawvalue {c in Counties};
param Fluvaccinationsrawvalue {c in Counties};
param Highschoolgraduationrawvalue {c in Counties};
param Somecollegerawvalue {c in Counties};
param Unemploymentrawvalue {c in Counties};
param Childreninpovertyrawvalue {c in Counties};
param Incomeinequalityrawvalue {c in Counties};
param Childreninsingleparenthouseholdsrawvalue {c in Counties};
param Injurydeathsrawvalue {c in Counties};
param Airpollutionparticulatematterrawvalue {c in Counties};
param Drivingalonetoworkrawvalue {c in Counties};
param Longcommutedrivingalonerawvalue {c in Counties};
param Lifeexpectancyrawvalue {c in Counties};
param Frequentphysicaldistressrawvalue {c in Counties};
param Frequentmentaldistressrawvalue {c in Counties};
param Diabetesprevalencerawvalue {c in Counties};
param Foodinsecurityrawvalue {c in Counties};
param Limitedaccesstohealthyfoodsrawvalue {c in Counties};
param Insufficientsleeprawvalue {c in Counties};
param Uninsuredadultsrawvalue {c in Counties};
param Uninsuredchildrenrawvalue {c in Counties};
param Medianhouseholdincomerawvalue {c in Counties};
param Childreneligibleforfreeorreducedpricelunchrawvalue {c in Counties};
param Homeownershiprawvalue {c in Counties};
param Populationrawvalue {c in Counties};
param below18yearsofagerawvalue {c in Counties};
param NonHispanicAfricanAmericanrawvalue {c in Counties};
param AmericanIndianandAlaskanNativerawvalue {c in Counties};
param Asianrawvalue {c in Counties};
param NativeHawaiianOtherPacificIslanderrawvalue {c in Counties};
param Hispanicrawvalue {c in Counties};
param NonHispanicwhiterawvalue {c in Counties};
param notproficientinEnglishrawvalue {c in Counties};
param Ruralrawvalue {c in Counties};

param benefit {c in Counties, i in Intervals} = 3.0058883227291653 * Poororfairhealthrawvalue[c] + -11.05854877787554 * Poorphysicalhealthdaysrawvalue[c] + -2.4009208101336235 * Poormentalhealthdaysrawvalue[c] + 3.3279128233855424 * Lowbirthweightrawvalue[c] + -1.805290660625115 * Adultsmokingrawvalue[c] + -2.4260241686832376 * Adultobesityrawvalue[c] + -12.29209926589423 * Foodenvironmentindexrawvalue[c] + 6.17739086132886 * Physicalinactivityrawvalue[c] + -1.2038162814229114 * Accesstoexerciseopportunitiesrawvalue[c] + 1.7565619934916672 * Excessivedrinkingrawvalue[c] + -0.3548003835144993 * Sexuallytransmittedinfectionsrawvalue[c] + 9.06924623296129 * Teenbirthsrawvalue[c] + 11.957524648071256 * Uninsuredrawvalue[c] + -1.479573383103534 * Primarycarephysiciansrawvalue[c] + -0.4267576164138748 * Dentistsrawvalue[c] + 1.5795237864513283 * Preventablehospitalstaysrawvalue[c] + -2.7966949919872732 * Mammographyscreeningrawvalue[c] + -0.6003619020295328 * Fluvaccinationsrawvalue[c] + 1.2354261151618022 * Highschoolgraduationrawvalue[c] + 0.1316107160411688 * Somecollegerawvalue[c] + 0.8418522960610177 * Unemploymentrawvalue[c] + 2.510690696640836 * Childreninpovertyrawvalue[c] + 2.0373569419350557 * Incomeinequalityrawvalue[c] + -0.5301040211974204 * Childreninsingleparenthouseholdsrawvalue[c] + 6.79677979484444 * Injurydeathsrawvalue[c] + -2.0622983815340183 * Airpollutionparticulatematterrawvalue[c] + -0.1431651906797038 * Drivingalonetoworkrawvalue[c] + -1.1136456745191206 * Longcommutedrivingalonerawvalue[c] + -75.85569534248941 * Lifeexpectancyrawvalue[c] + 15.169469869478137 * Frequentphysicaldistressrawvalue[c] + 2.9614493465559564 * Frequentmentaldistressrawvalue[c] + 4.526675445273829 * (Diabetesprevalencerawvalue[c] - i / 100000)+ -6.169799536035361 * Foodinsecurityrawvalue[c] + -5.104474325669326 * Limitedaccesstohealthyfoodsrawvalue[c] + -1.3726098317946842 * Insufficientsleeprawvalue[c] + -11.70949957582083 * Uninsuredadultsrawvalue[c] + -1.5982486575656605 * Uninsuredchildrenrawvalue[c] + 7.383161549677707 * Medianhouseholdincomerawvalue[c] + 3.375244498492169 * Childreneligibleforfreeorreducedpricelunchrawvalue[c] + -3.690943498898528 * Homeownershiprawvalue[c] + -0.08942691654758983 * Populationrawvalue[c] + -3.058767936636225 * below18yearsofagerawvalue[c] + -9.853101074845709 * NonHispanicAfricanAmericanrawvalue[c] + -2.7578541520831283 * AmericanIndianandAlaskanNativerawvalue[c] + -1.2093676666562478 * Asianrawvalue[c] + -0.4766173260835709 * NativeHawaiianOtherPacificIslanderrawvalue[c] + -11.120329851618104 * Hispanicrawvalue[c] + -12.513060155948105 * NonHispanicwhiterawvalue[c] + -1.9833853366815823 * notproficientinEnglishrawvalue[c] + -1.3385998495063678 * Ruralrawvalue[c] + 398.7224242424243;
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