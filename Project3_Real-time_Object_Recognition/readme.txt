compile 
gcc power.cpp  -o power

Discussion
In my program, power operation(2^2/2^100) is performed 10000000 times with pow and simple multiplication respectively.
Result 
10000000 times
2 power pow time = 0.114031
2 power simple multiplication time = 0.015627
100 power pow time = 0.124998
100 power simple multiplication time = 0.698987

According to the result, when the power is 2, simple multiplication is appearently faster than using pow function.
However when the power is 100, the pow function is much faster.
Thus, square two values is suitable to use simple multiplication. When the power is big enough, it's time to use pow function.