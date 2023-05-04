#include <matrixtypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100

using namespace ula;
using namespace std;

void Set_Data(Real A, Real B, RealVector &x_, Real x_max, RealVector &y_, Real E)
{
  srand(time(NULL));
  for (int k = 0; k < N; k++)
  {
    Real e_k = E * (rand() % 10000 - 5000.) / 5000.;
    x_(k) = x_max * (rand() % 10000) / 10000.;
    y_(k) = A * x_(k) * exp(-B * x_(k)) + e_k;
  }
}

Real Anzats(Real A, Real B, Real x)
{
  return A * x * exp(-B * x);
}

Real Error_function(Real A, Real B, RealVector &x_, RealVector &y_)
{
  Real S = 0.;
  for (int k = 0; k < N; k++)
    S += pow(Anzats(A, B, x_(k)) - y_(k), 2.);
  return S;
}

RealVector gradient(Real A, Real A_max, Real A_min,
                    Real B, Real B_max, Real B_min,
                    Real res, RealVector &x_, RealVector &y_)
{
  RealVector gradient_(2);
  gradient_(0) = (Error_function(A + (A_max - A_min) / res, B, x_, y_) - Error_function(A - (A_max - A_min) / res, B, x_, y_)) / (2. * (A_max - A_min) / res);
  gradient_(1) = (Error_function(A, B + (B_max - B_min) / res, x_, y_) - Error_function(A, B - (B_max - B_min) / res, x_, y_)) / (2. * (B_max - B_min) / res);
  return gradient_;
}

int main()
{
  // Set the data
  Real A_ = 1.24, B_ = 0.87, E = 0.05, x_max = 5.; // Data parameters
  RealVector x_(N), y_(N);                         // Data
  Set_Data(A_, B_, x_, x_max, y_, E);

  // Save the data in 'scatterred.dat'
  FILE *dskw;
  dskw = fopen("data/scattered.dat", "w+");
  for (int k = 0; k < N; k++)
    fprintf(dskw, "%g %g\n", x_(k), y_(k));
  fclose(dskw);

  // Save error parameter surface
  Real A_min = 0., A_max = 2., B_min = 0., B_max = 2., res = 200.;
  dskw = fopen("data/surface.dat", "w+");
  for (Real A = A_min; A < A_max; A += (A_max - A_min) / res)
  {
    for (Real B = B_min; B < B_max; B += (B_max - B_min) / res)
      fprintf(dskw, "%g %g %g\n", A, B, Error_function(A, B, x_, y_));
    fprintf(dskw, "\n");
  }
  fclose(dskw);

  // Calculate gradient vector field
  res = 50;
  dskw = fopen("data/gradient.dat", "w+");
  A_min = 0.;
  A_max = 2.;
  B_min = 0.;
  B_max = 2.;
  for (Real A = A_min; A < A_max; A += (A_max - A_min) / res)
    for (Real B = B_min; B < B_max; B += (B_max - B_min) / res)
    {
      RealVector gradient_(2);
      gradient_ = gradient(A, A_max, A_min, B, B_max, B_min, res, x_, y_);
      Real DA = -gradient_(0);
      Real DB = -gradient_(1);
      Real theta = atan2(DB, DA);
      fprintf(dskw, "%g %g %g %g %g\n",
              A - 0.5 * ((A_max - A_min) / res) * cos(theta), B - 0.5 * ((B_max - B_min) / res) * sin(theta),
              ((A_max - A_min) / res) * cos(theta), ((B_max - B_min) / res) * sin(theta), log(sqrt(DA * DA + DB * DB)));
    }
  fclose(dskw);

  // Gradient descent
  dskw = fopen("data/delta_function.dat", "w+");
  FILE *dskw2;
  dskw2 = fopen("data/minima.dat", "w+");
  for (int i = 0; i < 100; i++)
  {
    Real Ak = (rand() % 10000 / 10000.) * (A_max - A_min) + A_min;
    Real Bk = (rand() % 10000 / 10000.) * (B_max - B_min) + B_min;
    Real Ak_ = Ak, Bk_ = Bk;
    do
    {
      Ak = Ak_;
      Bk = Bk_;
      RealVector gradientk(2);
      gradientk = gradient(Ak, A_max, A_min, Bk, B_max, B_min, res, x_, y_);
      Real theta = atan2(-gradientk(1), -gradientk(0));
      Real dot = 1.;
      Real delta = 0.;
      do
      {
        delta += 0.0001;
        Ak_ = Ak + delta * cos(theta);
        Bk_ = Bk + delta * sin(theta);
        RealVector gradientk_(2);
        gradientk_ = gradient(Ak_, A_max, A_min, Bk_, B_max, B_min, res, x_, y_);
        dot = inner_prod(gradientk, gradientk_) / (inner_prod(gradientk, gradientk) * inner_prod(gradientk_, gradientk_));
        if (i < 8)
        {
          fprintf(dskw, "%g %g %g %g %d\n", Ak_, Bk_, Error_function(Ak_, Bk_, x_, y_), delta, i + 1);
          fflush(dskw);
        }
      } while (dot > 0.);
    } while (sqrt((Ak - Ak_) * (Ak - Ak_) + (Bk - Bk_) * (Bk - Bk_)) > 1e-4);
    fprintf(dskw2, "%g %g %g\n", Ak_, Bk_, Error_function(Ak_, Bk_, x_, y_));
  }
  fclose(dskw);
  fclose(dskw2);
  return 0;
}
