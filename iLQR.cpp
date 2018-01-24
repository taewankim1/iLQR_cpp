#include "iLQR.hpp"
#define tolFun 1e-3
#define tolGrad 1e-1
#define lambdaMax 1e8
#define lambdaMin 1e-6
#define PI 3.14159265
ofstream result_f;

double dist_circle(double x,double y,double * p,double r)
{
    double c = 0;
    c = sqrt( ( x - p[0] ) * ( x - p[0] ) + ( y - p[1] ) * ( y - p[1] ) ) - r; 
    return c;
};

iLQR::iLQR(double dt_temp)
{
    pathLeng = 1.2;
    pathR = 0.7;
    pathYoff = 1.2;
    x0 << 0,-pathLeng * 3 / 2 + 0.5 , 0;
    lambda = 1;
    dlambda = 1;
    lambdaFactor = 1.6;
    delT = dt_temp;
    x.resize(horizon+1);
    u.resize(horizon);
    xnew.resize(horizon+1);
    unew.resize(horizon);
    l.resize(horizon);
    L.resize(horizon);
    fx.resize(horizon);
    fu.resize(horizon);
    c.resize(horizon+1);
    cnew.resize(horizon+1);
    cx.resize(horizon+1);
    cu.resize(horizon);
    cxx.resize(horizon+1);
    cxu.resize(horizon);
    cuu.resize(horizon);
    Vx.resize(horizon+1);
    Vxx.resize(horizon+1);
    for(int i = 0; i < horizon; i++)
    {
        u[i].fill(0);
        x[i].fill(0);
        xnew[i].fill(0);
        unew[i].fill(0);

        l[i].fill(0);
        L[i].fill(0);

        fx[i].fill(0);
        fu[i].fill(0);

        // c[i].fill(0);
        // cnew[i].fill(0);

        c[i] = 0;
        cnew[i] = 0;

        cx[i].fill(0);
        cu[i].fill(0);

        cxx[i].fill(0);
        cxu[i].fill(0);
        cuu[i].fill(0);

        Vx[i].fill(0);
        Vxx[i].fill(0);

    }
    // c[horizon] = VectorXd::Zero(1);
    // cnew[horizon] = VectorXd::Zero(1);

    c[horizon] = 0;
    cnew[horizon] = 0;

    x[horizon] = VectorXd::Zero(dimX);
    xnew[horizon] = VectorXd::Zero(dimX);
    cx[horizon] = MatrixXd::Zero(dimX,1);
    cxx[horizon] = MatrixXd::Zero(dimX,dimX);

    Vx[horizon] = VectorXd::Zero(dimX);
    Vxx[horizon] = MatrixXd::Zero(dimX,dimX);


    Alpha.fill(1);
    // Alpha << 1,0.5012,0.2512,0.1259,0.0631,0.0316,0.0158,0.0079,0.004,0.002,0.001;
    VectorXd Alpha_pow = VectorXd::LinSpaced(11,0,-3);
    for( int i = 0; i < Alpha.rows(); i++)
    {
        Alpha(i) = pow( 10, Alpha_pow(i)) ;
    }
    dV = MatrixXd::Zero(1,2);
    // For numerical difference
    eps_x = MatrixXd::Identity(dimX,dimX);
    eps_u = MatrixXd::Identity(dimU,dimU);
    // For trace
    // jaehyun;
    for( int i = 0 ; i < maxIter; i++)
    {
        jaehyun.diff[i] = 0;
        jaehyun.backward[i] = 0;
        jaehyun.checkSmallGradient[i] = 0;
        jaehyun.forward[i] = 0;
        jaehyun.accept[i] = 0;
    }
    // Do you want a print?
    verbosity = false;
}

iLQR::~iLQR()
{

}

vector<Vector2d> iLQR::update(double X, double Y, double yaw, vector<Vector2d> u_init )
{
    // current position
    x0(0) = X;
    x0(1) = Y;
    x0(2) = yaw;
    
    // initial input
    u = u_init;

    // timer setting
    clock_t begin, end, begin_t, end_t;
    begin_t = clock();

    // initialize some parameter
    lambda = 1;
    dlambda = 1;
    lambdaFactor = 1.6;
    dV.fill(0);

    // parameter setting
    bool diverge = false;
    bool stop = false;

    // trace for iteration

    // timer, counters, constraints

    begin = clock();
    // initial trajectory
    x[0] = x0;
    for (int j = 0; j < this->Alpha.rows(); j++)
    {

        bool divergence_test = false;
        for(int i = 0; i < horizon; i++)
        {
            x[i+1] = this->dynamics(x[i],Alpha(j,0) *  u[i]);
            c[i] = estimate_cost(x[i], Alpha(j,0) * u[i]);
            if( x[i+1].array().abs().maxCoeff() > 1e8 )
            {
                diverge = true;
            }
        }
        c[horizon] = estimate_cost(x[horizon], VectorXd::Zero(dimU));
        // divergence test
        if ( diverge == false )
        {
            break;
        }

    }
    end = clock();
    // cout <<"Time for initial trajectory: " <<  (double)(end-begin) / CLOCKS_PER_SEC << endl;
    int iter = 0;
    bool flgChange = true;
    for( int iter_for = 0; iter_for < maxIter; iter_for++ )
    {

        if ( stop == true )
        {
            break;
        }
        iter++; 
        cout << iter << endl;
        begin = clock();
        // differentiate dynamics and cost
        xxu_double f_x_u;
        Matrix<double,dimX+dimU,1> c_x_u;
        Matrix<double,dimX+dimU,dimX+dimU> c_xx_uu;
        if ( flgChange == true )
        {
            for( int i = 0; i < horizon ; i++ )
            {
                f_x_u = get_dynamics_jacobian(x[i], u[i]);
                c_x_u = get_cost_jacobian(x[i], u[i]);
                c_xx_uu = get_cost_hessian(x[i],u[i]);
                c_xx_uu = 0.5 * ( c_xx_uu + c_xx_uu.transpose() );
                fx[i] = f_x_u.topLeftCorner(dimX,dimX);
                fu[i] = f_x_u.bottomRightCorner(dimX,dimU);
                cx[i] = c_x_u.topRows(dimX);
                cu[i] = c_x_u.bottomRows(dimU);
                cxx[i] = c_xx_uu.topLeftCorner(dimX,dimX);
                cxu[i] = c_xx_uu.topRightCorner(dimX,dimU);
                cuu[i] = c_xx_uu.bottomRightCorner(dimU,dimU);
            }
            c_x_u = get_cost_jacobian(x[horizon],MatrixXd::Zero(dimU,1));
            c_xx_uu = get_cost_hessian(x[horizon],MatrixXd::Zero(dimU,1));
            c_xx_uu = 0.5 * ( c_xx_uu + c_xx_uu.transpose() );
            cx[horizon] = c_x_u.topRows(dimX);
            cxx[horizon] = c_xx_uu.topLeftCorner(dimX,dimX);
            flgChange = false;
        }
        end = clock();
        jaehyun.diff[iter-1] = (double)(end-begin) / CLOCKS_PER_SEC; 

        // backward pass
        bool backPassDone = false;
        while (backPassDone == false)
        {
            begin = clock();
            diverge = backward();
            end = clock();
            jaehyun.backward[iter-1] = (double)(end-begin) / CLOCKS_PER_SEC; 

            if (diverge == true)
            {
                cout << "Cholesky failed" << endl;
                if( lambda * lambdaFactor >= lambdaFactor )
                {   
                    dlambda = lambda*lambdaFactor;
                }
                else
                {
                    dlambda = lambdaFactor;
                }
                if ( lambda * dlambda > lambdaMin )
                {
                    lambda = lambda*dlambda;
                }
                else
                {
                    lambda = lambdaMin;
                }
                if ( lambda > lambdaMax )
                { 
                    break;
                }
            }
            else
            {
                backPassDone = true;
            }
        }

        begin = clock();
        // check for termination due to small gradient
        double g_norm_sum = 0;
        double g_norm = 0;
        double g_1;
        double g_2;
        for( int i = 0 ; i < horizon; i++ )
        {
            g_1 = abs( l[i](0,0) ) / ( abs( u[i](0,0) ) + 1 );
            g_2 = abs( l[i](1,0) ) / ( abs( u[i](1,0) ) + 1 );
            if( g_1 >= g_2 )
            {
                g_norm_sum = g_norm_sum + g_1;
            }
            else
            {
                g_norm_sum = g_norm_sum + g_2;
            }

        }
        g_norm = g_norm_sum / horizon;
        if ( g_norm < tolGrad && lambda < 1e-5)
        {
            if( dlambda / lambdaFactor <= 1 / lambdaFactor )
            {   
                dlambda = dlambda / lambdaFactor;
            }
            else
            {
                dlambda = 1 / lambdaFactor;
            }
            if ( lambda  > lambdaMin )
            {
                lambda = lambda*dlambda;
            }
            cout << "SUCCEESS: gradient norm < tolGrad" << endl;
            break;
        }
        end = clock();
        jaehyun.checkSmallGradient[iter-1] =  (double)(end-begin) / CLOCKS_PER_SEC; 

        begin = clock();
        // forward pass(serial line search) 
        bool fwdPassDone = false;
        double alpha_temp;
        double dcost = 0;
        double expected = 0;
        double z = 0;
        if (backPassDone)
        {
            for(int i = 0; i < Alpha.rows(); i++)
            {
                alpha_temp = Alpha(i,0);
                forward(x[0], u, L, x, l, alpha_temp);
                dcost = 0;
                for(int j = 0; j < horizon+1; j++)
                {
                    dcost = dcost + c[j] - cnew[j];
                }
                expected = -alpha_temp * ( dV(0,0) + alpha_temp * dV(0,1));
                if ( expected > 0 )
                {
                    z = dcost / expected;
                }
                else
                {
                    z = dcost / abs(dcost);
                    cout << "non-positive expected reduction: should not occur" << endl;
                }
                if ( z > 0 )
                {
                    fwdPassDone = true;
                    break;
                }

            }
            if ( fwdPassDone == false)
            {
                alpha_temp = 1e8;
            }
        }
        end = clock();
        jaehyun.forward[iter-1] =  (double)(end-begin) / CLOCKS_PER_SEC; 

        begin = clock();
        // accept step
        if (fwdPassDone)
        {
            // decrese lambda
            if( lambda / lambdaFactor <= 1 / lambdaFactor )
            {   
                dlambda = lambda / lambdaFactor;
            }
            else
            {
                dlambda = 1 / lambdaFactor;
            }
            if ( lambda  > lambdaMin )
            {
                lambda = lambda*dlambda;
            }
            // accept changes
            u = unew;
            x = xnew;
            c = cnew;
            flgChange = true;
            // terminate ?
            if ( dcost < tolFun )
            {
                cout << "SUCCESS: cost change < tolFun" << endl;
                break;
            }
        }
        else // no cost improvement
        {
            // increse lambda
            if( lambda * lambdaFactor >= lambdaFactor )
            {   
                dlambda = lambda*lambdaFactor;
            }
            else
            {
                dlambda = lambdaFactor;
            }
            if ( lambda * dlambda > lambdaMin )
            {
                lambda = lambda*dlambda;
            }
            else
            {
                lambda = lambdaMin;
            }
            // terminate ?
            if ( lambda > lambdaMax )
            {
                cout << "EXIT: lambda > lambdaMax" << endl;
                break;
            }

        }
        end = clock();
        jaehyun.accept[iter-1] =  (double)(end-begin) / CLOCKS_PER_SEC; 

        if( iter == maxIter )
        {
            cout << "EXIT: Maximum iteration reached" << endl;
        }
        // print?
        if ( verbosity == true )
        {
            cout <<"Time for differentiate: " <<  jaehyun.diff[iter-1] << endl;
            cout <<"Time for backward: " <<  jaehyun.backward[iter-1] << endl;
            cout <<"Time for checkSmallGradient: " <<  jaehyun.checkSmallGradient[iter-1]  << endl;
            cout <<"Time for forward: " <<  jaehyun.forward[iter-1] << endl;
            cout <<"Time for accept: " << jaehyun.accept[iter-1]   << endl;

        }

    }
    // end
    end_t = clock();
    // record the results;
    MatrixXd x_out = MatrixXd::Zero(dimX,horizon+1);
    MatrixXd u_out = MatrixXd::Zero(dimU,horizon);
    MatrixXd c_out = MatrixXd::Zero(1,horizon+1);
    for (int i = 0; i < horizon+1 ; i++)
    {
        x_out.col(i) = x[i];
        c_out.col(i) << c[i];
        if( i < horizon )
        {
            u_out.col(i) = u[i];
        }
    }
//     result_f.open("/home/keemtw/catkin_ws/src/unicycle_simulation/log/x_result.txt");
//     result_f << x_out;
//     result_f.close();
//     result_f.open("/home/keemtw/catkin_ws/src/unicycle_simulation/log/u_result.txt");
//     result_f << u_out;
//     result_f.close();
//     result_f.open("/home/keemtw/catkin_ws/src/unicycle_simulation/log/c_result.txt");
//     result_f << c_out;
//     result_f.close();

    cout <<"Computational time: " <<  (double)(end_t -begin_t ) / CLOCKS_PER_SEC << endl;
    cout <<"Time/iter: " << (double)(end_t -begin_t) / CLOCKS_PER_SEC / iter << endl;
    return u;
}

void iLQR::forward(Vector3d state0, vector<Vector2d> input, vector<feedback_gain> K, vector<Vector3d> state, vector<Vector2d> k, double alpha_temp )
{
    Vector3d dx;

    xnew[0] = state0;
    for( int i = 0 ; i < horizon ; i++ )
    {
        dx = xnew[i] - state[i];
        unew[i] = input[i] + k[i]*alpha_temp + K[i] * dx;
        xnew[i+1] = this->dynamics( xnew[i], unew[i] );
        cnew[i] = this->estimate_cost( xnew[i], unew[i] );

    }


    cnew[horizon] = estimate_cost(xnew[horizon], Vector2d::Zero(dimU) );
}

bool iLQR::backward()
{
    bool diverge = false;

    // MatrixXd dV = MatrixXd::Zero(1,2);

    Vx[horizon] = cx[horizon];
    Vxx[horizon] = cxx[horizon];

    Matrix<double,dimU,1> Qu = MatrixXd::Zero(dimU,1);
    Matrix<double,dimX,1> Qx = MatrixXd::Zero(dimX,1);
    Matrix<double,dimU,dimX> Qux = MatrixXd::Zero(dimU,dimX);
    uu_double Quu = MatrixXd::Zero(dimU,dimU);
    xx_double Qxx = MatrixXd::Zero(dimX,dimX);

    xx_double Vxx_reg = MatrixXd::Zero(dimX,dimX);
    Matrix<double,dimU,dimX> Qux_reg = MatrixXd::Zero(dimU,dimX);
    uu_double QuuF = MatrixXd::Zero(dimU,dimU);

    Matrix<double,dimU,1> k_i = MatrixXd::Zero(dimU,1);
    Matrix<double,dimU,dimX> K_i = MatrixXd::Zero(dimU,dimX);

    dV.fill(0);
    bool diverge_test = false;
    for ( int i = horizon - 1; i >= 0 ; i-- )
    {
        Qu = cu[i] + fu[i].transpose() * Vx[i+1];
        Qx = cx[i] + fx[i].transpose() * Vx[i+1];

        Qux = cxu[i].transpose() + fu[i].transpose() * Vxx[i+1] * fx[i];


        Quu = cuu[i] + fu[i].transpose() * Vxx[i+1] * fu[i];

        Qxx = cxx[i] + fx[i].transpose() * Vxx[i+1] * fx[i];

        Vxx_reg = Vxx[i+1] + lambda * MatrixXd::Identity(dimX,dimX);

        Qux_reg = cxu[i].transpose() + fu[i].transpose() * Vxx_reg * fx[i];

        QuuF = cuu[i] + fu[i].transpose() * Vxx_reg * fu[i];

//
//        LLT<MatrixXd> LLT_y(QuuF);
//        Lower_mat = LLT_y.matrixL();
//        int p = ( (Lower_mat*0).eval().nonZeros() == 0 );
//        if ( p != 0 )
//        {
//            diverge_test = true;
//            cout << "backward was diverged" << endl;
//            return diverge_test;
//        }
//        k_i = -Lower_mat.inverse() * Lower_mat.transpose().inverse() * Qu;
//        K_i = -Lower_mat.inverse() * Lower_mat.transpose().inverse() * Qux_reg;
        k_i = -QuuF.inverse() * Qu;
        K_i = -QuuF.inverse() * Qux_reg;
        // update cost-to-go approximation
        dV.col(0) = k_i.transpose() * Qu+dV.col(0); 
        dV.col(1) = 0.5 * k_i.transpose() * Quu * k_i+dV.col(1) ;
        Vx[i] = Qx + K_i.transpose() * Quu * k_i + K_i.transpose() * Qu + Qux.transpose() * k_i;
        Vxx[i] = Qxx + K_i.transpose() * Quu * K_i + K_i.transpose() * Qux + Qux.transpose() * K_i;
        Vxx[i] = 0.5 * ( Vxx[i] + Vxx[i].transpose() );
        // save controls/gains
        l[i] = k_i;
        L[i] = K_i;
    }
    return diverge_test;

}



Vector3d iLQR::dynamics(Vector3d state, Vector2d input)
{
    // state
    double & X = state(0);
    double & Y = state(1);
    double & yaw = state(2);
    
    // controls
    double & v = input(0);
    double & w = input(1);
    
    // Ideal differential dirivng model
    Vector3d f;
    f(0) = v * cos(yaw);
    f(1) = v * sin(yaw);
    f(2) = w;


    Vector3d y = MatrixXd::Zero(3,1);
    y = state + delT * f;

    return y;
}

xxu_double iLQR::get_dynamics_jacobian(Vector3d state, Vector2d input)
{
    double h = pow(2,-17);
    
    xxu_double J = MatrixXd::Zero(dimX,dimX+dimU);

    // Numerical difference
    for ( int i = 0; i < dimX; i++ )
    {
        J.col(i) = ( dynamics( state + h * eps_x.col(i) , input ) - dynamics(state,input) ) / h; 
    }
    for (int i = 0; i < dimU; i++ )
    {
        J.col(i+3) =  ( dynamics( state , input + h*eps_u.col(i) ) - dynamics(state,input) ) / h;
    }

    return J;
}


double iLQR::estimate_cost(Vector3d state, Vector2d input)
{
    // state
    double & X = state(0);
    double & Y = state(1);
    double & yaw = state(2);

    // controls
    double & v = input(0);
    double & w = input(1);

    // cost
    // double lc = 10*v*v +10*w*w;
    // double lx = X*X + Y*Y + yaw*yaw;
    // return lc + lx;

    // cost for path
    // cost for input
    double lcu = 0.7 * w * w;
    // cost got during running
    // double lx = 5 * pow( hFunction(X,Y,pathLeng, pathR, pathYoff), 2);
    double lx = 5 * pow( hFunctionTr(X,Y,1.8, 0.5, 2.4), 2);
    // cpst for velocity
    double lv = 5 * (1 - v) * (1 - v);
    return lcu + lx + lv;

}

xu_vec_double iLQR::get_cost_jacobian(Vector3d state, Vector2d input)
{
    double h = pow(2,-17);
    // double h = 1e-5;

    xu_vec_double J;

    for ( int i = 0; i < dimX; i++ )
    {
        J(i,0) = ( estimate_cost( state + h * eps_x.col(i) , input ) - estimate_cost(state,input) ) / h; 
    }
    for (int i = 0; i < dimU; i++ )
    {
        J(i+3,0) = ( estimate_cost( state , input + h*eps_u.col(i) ) - estimate_cost(state,input) ) / h;
    }

    return J;

}

xxuu_double iLQR::get_cost_hessian(Vector3d state, Vector2d input)
{
    double h = pow(2,-17);

    double d1 = 0;
    double d0 = 0;

    xxuu_double J;

    for ( int i = 0; i < dimX; i++ )
    {
        J.col(i) = ( get_cost_jacobian( state + h * eps_x.col(i) , input ) - get_cost_jacobian(state,input) ) / h; 
    }
    for (int i = 0; i < dimU; i++ )
    {
        J.col(i+3) =  ( get_cost_jacobian( state , input + h*eps_u.col(i) ) - get_cost_jacobian(state,input) ) / h;
    }
    return J;

}

double iLQR::hFunction(double X, double Y, double l, double r, double yoff)
{
    double p1_x = 0;
    double p1_y = yoff;
    double p2_x = 0;
    double p2_y = -yoff;
    double c = 0;
    double distanceToP1;
    double distanceToP2;
    double xVal;
    if( Y > yoff )
    {
        distanceToP1 = sqrt( ( X - p1_x ) * ( X - p1_x ) + ( Y - p1_y ) * ( Y - p1_y ) ); 
        c = ( distanceToP1 - l ) / ( r );
    }
    else if ( Y < -yoff )
    {
        distanceToP2 = sqrt( ( X - p2_x ) * ( X - p2_x ) + ( Y - p2_y ) * ( Y - p2_y ) );
        c = ( distanceToP2 - l ) / ( r );
    }
    else
    {
        xVal = abs(X);
        c = ( xVal - l ) / ( r ) ;
    }
    if ( c > 1 )
    {
        c = 1;
    }
    else if ( c < -1 )
    {
        c = -1;
    }

    return c;
}

double iLQR::hFunctionSq(double X, double Y, double l, double r, double yoff)
{
    double p1[2] = {l-r,yoff-r};
    double p2[2] = {l+r,yoff+r};
    double p3[2] = {-l-r,yoff+r};
    double p4[2] = {-l+r,yoff-r};
    double p5[2] = {-l+r,l/2+r};
    double p6[2] = {-l-r,l/2-r};
    double p7[2] = {+r,l/2+r};
    double p8[2] = {-r,l/2-r};
    double p9[2] = {-r,-l/2+r};
    double p10[2] = {+r,-l/2-r};
    double p11[2] = {-l+r,-l/2-r};
    double p12[2] = {-l-r,-l/2+r};
    double p13[2] = {-l-r,-yoff-r};
    double p14[2] = {-l+r,-yoff+r};
    double p15[2] = {l-r,-yoff+r};
    double p16[2] = {l+r,-yoff-r};
    double c = 0;
    if ( ( X <= p2[0] ) & ( X >= p1[0] ) & ( Y < p1[1] ) & ( Y >= p15[1] ) )
    {
        c = (X-l) / r;
    }
    else if( ( X <= p2[0] ) & ( X >= p1[0] ) & ( Y <= p2[1] ) & ( Y > p1[1] ) )
    {
        c = dist_circle(X,Y,p1,r) / r;
    }
    else if( ( X < p1[0] ) & ( X >= p4[0] ) & ( Y <= p2[1] ) & ( Y >= p1[1] ) )
    {
        c = (Y-yoff) / r;
    }
    else if( ( X < p4[0] ) & ( X >= p3[0] ) & ( Y <= p3[1] ) & ( Y >= p4[1] ) )
    {
        c = dist_circle(X,Y,p4,r) / r;
    }
    else if( ( X <= p4[0] ) & ( X >= p3[0] ) & ( Y < p4[1] ) & ( Y >= p5[1] ) )
    {
        c = -(X- (-l) ) / r;
    }
    else if( ( X < p5[0] ) & ( X >= p6[0] ) & ( Y <= p5[1] ) & ( Y >= p6[1] ) )
    {
        c = dist_circle(X,Y,p5,r) / r;
    }
    else if( ( X < p8[0] ) & ( X >= p5[0] ) & ( Y <= p5[1] ) & ( Y >= p8[1] ) )
    {
        c = -(Y-(l/2)) / r;
    }
    else if( ( X <= p7[0] ) & ( X >= p8[0] ) & ( Y <= p7[1] ) & ( Y > p8[1] ) )
    {
        c = -dist_circle(X,Y,p8,r) / r;
    }
    else if( ( X <= p7[0] ) & ( X >= p8[0] ) & ( Y <= p8[1] ) & ( Y > p9[1] ) )
    {
        c = -X/r;
    }
    else if( ( X <= p10[0] ) & ( X > p9[0] ) & ( Y <= p9[1] ) & ( Y >= p10[1] ) )
    {
        c = -dist_circle(X,Y,p9,r) / r;
    }
    else if( ( X <= p9[0] ) & ( X > p11[0] ) & ( Y <= p9[1] ) & ( Y >= p10[1] ) )
    {
        c = (Y-(-l/2))/r;
    }
    else if( ( X <= p11[0] ) & ( X >= p12[0] ) & ( Y <= p12[1] ) & ( Y > p11[1] ) )
    {
        c = dist_circle(X,Y,p11,r) / r;
    }
    else if( ( X <= p11[0] ) & ( X >= p12[0] ) & ( Y <= p11[1] ) & ( Y > p14[1] ) )
    {
        c = -(X-(-l))/r;
    }
    else if( ( X <= p14[0] ) & ( X >= p13[0] ) & ( Y <= p14[1] ) & ( Y > p13[1] ) )
    {
        c = dist_circle(X,Y,p14,r) / r;
    }
    else if( ( X < p15[0] ) & ( X >= p14[0] ) & ( Y <= p14[1] ) & ( Y > p13[1] ) )
    {
        c = -(Y-(-yoff)) / r;
    }
    else if( ( X <= p16[0] ) & ( X >= p15[0] ) & ( Y <= p15[1] ) & ( Y >= p16[1] ) )
    {
        c = dist_circle(X,Y,p15,r) / r;
    }
    else if( ( X > p2[0] ) | ( X < p3[0] ) | ( Y > p2[1] ) | ( Y < p16[1] ) | ( 
             ( X < p8[0] ) & ( X >= p6[0] ) & ( Y < p8[1] ) & ( Y > p9[1] ) ) )
    {
        c = 1;
    }
    else
    {
        c = -1;
    }


    return c;
}

double iLQR::hFunctionTr(double X, double Y, double l, double r, double yoff)
{
    double p1[2] = {l-r,yoff-r};
    double p2[2] = {l+r,yoff+r};
    double p3[2] = {-l-r,yoff+r};
    double p4[2] = {-l+r,yoff-r};
    double p5[2] = {-l+r,l/2+r};
    double p6[2] = {-l-r,l/2-r};
    double p7[2] = {+r,l/2+r};
    double p8[2] = {-r,l/2-r};
    double p9[2] = {-r,-l/2+r};
    double p10[2] = {+r,-l/2-r};
    double p11[2] = {-l+r,-l/2-r};
    double p12[2] = {-l-r,-l/2+r};
    double p13[2] = {-l-r,-yoff-r};
    double p14[2] = {-l+r,-yoff+r};
    double p15[2] = {l-r,-yoff+r};
    double p16[2] = {l+r,-yoff-r};
    double c = 0;
    double pi = 3.141592;
    double theta1 = pi - atan(yoff/l);
    double theta2 = 1.5*pi - theta1;
    double t1 = atan2( Y - p4[1], X - p4[0] );
    double t2 = atan2( Y - p15[1], X - p15[0] );

    if ( ( X >= p1[0] ) & ( Y < p1[1] ) & ( Y >= p15[1] ) )
    {
        c = (X-l) / r;
    }
    else if( ( X >= p1[0] ) & ( Y > p1[1] ) )
    {
        c = dist_circle(X,Y,p1,r) / r;
    }
    else if( ( X < p1[0] ) & ( X >= p4[0] ) & ( Y >= p1[1] ) )
    {
        c = (Y-yoff) / r;
    }
    else if( ( t1 >= pi/2 ) | ( t1 < -(2*pi-theta1-pi/2) ) )
    {
        c = dist_circle(X,Y,p4,r) / r;
    }
    else if( ( t2 < 0 ) & ( t2 > -theta2 ) )
    {
        c = dist_circle(X,Y,p15,r) / r;
    }
    else if( ( t1 <= -(pi-theta1) ) & ( t1 >= -(2*pi-theta1-pi/2) ) 
                & ( ( t2 >= pi/2+pi - theta2 ) | ( t2 <= -theta2 ) ) )
    {
        c = ( abs( (p15[1] - p4[1] ) * X - (p15[0] -p4[0]) * Y + p15[0] * p4[1] - p15[1] * p4[0] )
                / sqrt( (p15[1] - p4[1])*(p15[1] - p4[1]) + (p15[0] - p4[0])*(p15[0] - p4[0]) )
                - r) / r;
    }
    else
    {
        c = -1;
    }
    if ( c > 1 )
    {
        c = 1;
    }
    else if( c < -1 )
    {
        c = -1;
    }
    return c;
}

// Another way to calculate the hessian.
// I recorded this method in here.
//     for( int i = 0; i < dimX; i++ )
//     {
//         for (int j = 0 ; j < dimX; j++)
//         {
//            d1 = ( estimate_cost(state + h * eps_x.col(i) + h * eps_x.col(j), input) - 
//                    estimate_cost(state + h * eps_x.col(j), input ) ) / h;
//            d0 = ( estimate_cost(state + h * eps_x.col(i),input ) - estimate_cost(state , input ) ) / h; 
//            J(j,i) = ( d1 - d0 ) / h;            
//         }
//         for (int j = dimX; j < dimX+dimU; j++)
//         {
//            d1 = ( estimate_cost(state + h * eps_x.col(i), input + h * eps_u.col(j-dimX) ) - 
//                    estimate_cost(state, input + h * eps_u.col(j-dimX) ) ) / h;
//            d0 = ( estimate_cost(state + h * eps_x.col(i),input ) - estimate_cost(state , input ) ) / h; 
//            J(j,i) = ( d1 - d0 ) / h;            
//         }
//     }
//     for( int i = dimX; i < dimX+dimU ; i++ )
//     {
//         for (int j = 0 ; j < dimX; j++)
//         {
//            d1 = ( estimate_cost(state  + h * eps_x.col(j), input + h * eps_u.col(i-dimX) ) - 
//                    estimate_cost(state + h * eps_x.col(j), input ) ) / h;
//            d0 = ( estimate_cost(state ,input + h*eps_u.col(i-dimX) ) - estimate_cost(state , input ) ) / h; 
//            J(j,i) = ( d1 - d0 ) / h;            
//             
//         }
//         for (int j = dimX; j < dimX+dimU; j++)
//         {
//            d1 = ( estimate_cost(state, input + h * eps_u.col(j-dimX) + h * eps_u.col(i-dimX) ) - 
//                    estimate_cost(state, input + h * eps_u.col(j-dimX) ) ) / h;
//            d0 = ( estimate_cost(state ,input + h*eps_u.col(i-dimX) ) - estimate_cost(state , input ) ) / h; 
//            J(j,i) = ( d1 - d0 ) / h;            
// 
//         }
//     }


