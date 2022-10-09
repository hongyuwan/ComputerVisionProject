#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>

using namespace std;

double getTime()
{
  struct timeval cur;

  gettimeofday(&cur, NULL);
  return (cur.tv_sec + cur.tv_usec / 1000000.0);
}

int main()
{
  int p = 2;
  int n = 2;
  double start_timep, end_timep, timep, start_time, end_time, time;
  double start_timep1, end_timep1, timep1, start_time1, end_time1, time1;
  start_timep = getTime();
  for (int i = 0; i < 10000000; i++)
  {
    pow(n, 2);
  }
  end_timep = getTime();
  timep = end_timep - start_timep;
  start_time = getTime();
  for (int j = 0; j < 10000000; j++)
  {
    for (int i = 0; i < 1; i++)
    {
      p = p * n;
    }
  }
  end_time = getTime();
  time = end_time - start_time;
  //*********multi power*********//
  start_timep1 = getTime();
  for (int i = 0; i < 10000000; i++)
  {
    pow(n, 100);
  }
  end_timep1 = getTime();
  timep1 = end_timep1 - start_timep1;
  start_time1 = getTime();
  for (int j = 0; j < 10000000; j++)
  {
    for (int i = 0; i < 99; i++)
    {
      p = p * n;
    }
  }
  end_time1 = getTime();
  time1 = end_time1 - start_time1;
  printf("2 power pow time = %f\n", timep);
  printf("2 power simple multiplication time = %f\n", time);
  printf("100 power pow time = %f\n", timep1);
  printf("100 power simple multiplication time = %f\n", time1);
}