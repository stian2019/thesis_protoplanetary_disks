/* /////////////////////////////////////////////////////////////////// */
/*! \file
 *  \brief Specification of explicit first and second viscosity coefficients*/
/* /////////////////////////////////////////////////////////////////// */
#define MAX(a, b) ((a) > (b) ? a : b)
#define MIN(a, b) ((a) < (b) ? a : b)

#include "pluto.h"
#include <math.h>

double Au = 14959787070000.0;
double GM = 1.3271244e+26;
double Sigma_0 = 6.;
double x1_value[300],LastAlpha[300];
int inited=  0, indexed=0;
double slope=3.e-16;
/* ************************************************************************** */
void Visc_nu(double *v, double x1, double x2, double x3, double *nu1,
             double *nu2)
/*!
 *
 *  \param [in]      v  pointer to data array containing cell-centered
 * quantities \param [in]      x1 real, coordinate value \param [in]      x2
 * real, coordinate value \param [in]      x3 real, coordinate value \param [in,
 * out] nu1  pointer to first viscous coefficient \param [in, out] nu2  pointer
 * to second viscous coefficient
 *
 *  \return This function has no return value.
 * ************************************************************************** */
{


        // double y;
        // if((v[RHO] - 15.)>=0){
        //   y = 1. - 0.5 * exp(-(v[RHO] - 15.) / 30.);
        //}
        // else{
        //  y = 0.5 * exp(MIN((v[RHO] - 15.) / 30., 709.7));
        //}

        // double alpha=1e-2 - (1e-2 - 1e-4) * y;




        double Rho_0= Sigma_0 * pow((x1 / (100. * Au)), -1);
        double alpha = (1 - tanh((v[RHO] - 15.) / (1.) * 0.2)) * 2.e-2 / 2. + 1.e-4;
        // double alpha=0.02;

        if(indexed<300)
        {
                x1_value[indexed]=x1;
                LastAlpha[indexed]=alpha;
                indexed++;
        }else
        {
                for(int ii=0; ii<300; ii++)
                {
                        //printf("%d,%fAU\n", ii,x1_value[ii]/Au);
                        if(fabs(x1_value[ii]-x1)<1.e-6*x1)
                        {
                                if(alpha>LastAlpha[ii])
                                {
                                        alpha=MIN(alpha,LastAlpha[ii]+g_dt*slope);
                                        LastAlpha[ii]=alpha;
                                        //printf("%d,%f,%fU\n",ii,x1/Au,alpha);
                                }else
                                {
                                        alpha=MAX(alpha,LastAlpha[ii]-g_dt*slope);
                                        LastAlpha[ii]=alpha;
                                        //printf("%d,%f,%fD\n",ii,x1/Au,alpha);
                                }
                        }
                }
        }


        double Omega = v[VX2] / x1;
        double H = 0.05 * x1;
        double cs = H * Omega;
        double etkv = alpha * cs * H; // effective turbulent kinematic viscosity
        *nu1 = v[RHO] * etkv;
        *nu2 = 0.0;
}
