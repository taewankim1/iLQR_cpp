#ifndef ILQR_H_
#define ILQR_H_
#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Cholesky>
#include <fstream>
#include <time.h>
#define dimX 3
#define dimU 2
#define horizon 50
#define maxIter 5000
// #include <CXX11/Tensor>
// #include <vector>
// #include <unsupported/Eigen/CXX11/Tensor>
using namespace Eigen;
using namespace std;
typedef Matrix<double,dimX+dimU,1> xu_vec_double;
typedef Matrix<double,dimX,dimX> xx_double;
typedef Matrix<double,dimX,dimU> xu_double;
typedef Matrix<double,dimU,dimU> uu_double;
typedef Matrix<double,dimU,dimX> feedback_gain;
typedef Matrix<double,dimX,dimX+dimU> xxu_double;
typedef Matrix<double,dimX+dimU,dimX+dimU> xxuu_double;
typedef Matrix<double,1,1> c_double;

struct trace{
    double diff[maxIter] ;
    double backward[maxIter];
    double checkSmallGradient[maxIter];
    double forward[maxIter];
    double accept[maxIter];

};
class iLQR
{
    public : 

        iLQR(double dt_temp );
        ~iLQR();

        vector<Vector2d> update( double X, double Y, double yaw , vector<Vector2d> u_init );
        void forward(Vector3d state0, vector<Vector2d> input, vector<feedback_gain> K, vector<Vector3d> state, vector<Vector2d> k,double alpha_temp  );
        bool backward( );
        void initialize();
        Vector3d dynamics(Vector3d state, Vector2d input);
        xxu_double get_dynamics_jacobian(Vector3d state, Vector2d input);
        double estimate_cost(Vector3d state, Vector2d input);
        xu_vec_double get_cost_jacobian(Vector3d state, Vector2d input);
        xxuu_double get_cost_hessian(Vector3d state, Vector2d input);
        double hFunction(double X, double Y, double l, double r, double yoff);
        double hFunctionSq(double X, double Y, double l, double r, double yoff);
        double hFunctionTr(double X, double Y, double l, double r, double yoff);
    private :
       
        bool verbosity;
        trace jaehyun;

        double pathLeng;
        double pathR;
        double pathYoff;
        double dlambda;
        double lambda;
        double lambdaFactor;
        double delT;
        
        xx_double eps_x;
        uu_double eps_u;

        Matrix<double,1,2> dV;
        Vector3d x0;
        vector<Vector3d> x;
        vector<Vector2d> u;
        vector<Vector3d> xnew;
        vector<Vector2d> unew;
        Matrix<double,11,1> Alpha;
        vector<Vector2d> l;
        vector<feedback_gain> L;
        vector<xx_double, aligned_allocator<xx_double> > fx;
        vector<xu_double, aligned_allocator<xu_double> > fu;
        vector<double> c;
        vector<double> cnew;
        vector<Vector3d, aligned_allocator<Vector3d> > cx;
        vector<Vector2d, aligned_allocator<Vector2d> > cu;
        vector<xx_double, aligned_allocator<xx_double> > cxx;
        vector<xu_double, aligned_allocator<xu_double> > cxu;
        vector<uu_double, aligned_allocator<uu_double> > cuu;
        vector<Vector3d, aligned_allocator<Vector3d>  > Vx;
        vector<xx_double, aligned_allocator<xx_double> > Vxx;

        // spGP sangil;
};







#endif /* */
