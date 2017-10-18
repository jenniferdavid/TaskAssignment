/************************************************/
/*    Task_Assignment.cc                               */
/************************************************/

#include <fstream>
#include <math.h>
#include <iomanip> // needed for setw(int)
#include <string>
#include "stdio.h"
#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/LU> 
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <time.h>

using namespace Eigen;
using namespace std;

int nTasks = 1;
int nVehicles = 1;
int nDim = 2*nVehicles + nTasks;
double kT_start = 10.0;
double kT_stop  = 0.001;
double kT_swfac = 0.95;
double kT_fac = exp( log(kT_swfac) / (nVehicles * nDim) ); // lower T after every neuron update
double kT = kT_start * kT_fac;  // * to give the ini conf a own datapoint in sat plot
double Elocal, E_loop, E_assign;
double Pji_coeff = .001;
            
Eigen::MatrixXd VMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd DeltaMatrix = MatrixXd::Ones(nDim,nDim);
Eigen::MatrixXd TauMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd vDeltaR;
Eigen::MatrixXd vDeltaL;
Eigen::MatrixXd sumv = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd sump = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd PMatrix;
Eigen::MatrixXd I = MatrixXd::Identity(nDim,nDim); //identity matrix
Eigen::VectorXd TVec(nDim);
Eigen::VectorXd vdVecL(nDim);
Eigen::VectorXd vdVecR(nDim);
Eigen::VectorXd rightVec(nDim);
Eigen::VectorXd leftVec(nDim);
Eigen::VectorXd valVec1(nVehicles);
Eigen::VectorXd valVec2(nVehicles);
Eigen::VectorXd valVec(nVehicles+nVehicles);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void getVMatrix(int nVehicles,int nTasks)
    {
        cout << "\n nDim is: " << nDim << endl;
        double tmp = 1./nDim;
        cout << "\n tmp is: " << tmp << endl;
        for (int i = 0; i < nDim; i++)
        {
            for (int j = 0; j < nDim; j++)
            {
                VMatrix(i,j) = tmp + 0.02*(rand() - 0.5);	// +-1 % noise
            }
        }
    
    VMatrix.diagonal().array() = 0;
    VMatrix.leftCols(nVehicles) *= 0;
    VMatrix.bottomRows(nVehicles) *= 0;
    VMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  //adding all the constraints for the vehicles and tasks

    return;
    }
    
////////////////////////////////////////////////////////////////////////////////////////
   
     void update(Eigen::MatrixXd VMatrix,Eigen::MatrixXd PMatrix)
{
       double min_Etest = 500;
       double Etest;
       cout << "\n Etest is " << min_Etest << endl;
       cout << "\n Elocal is " << Elocal << endl;
       cout << "\n kT is " << kT << endl;
       cout << "nr is: " << min_Etest-Elocal << endl;
       double arg = ((min_Etest-Elocal)/kT);
       cout << "\n arg value is: " << arg << endl;
       cout << "\n exp(arg) value is: " << exp(arg) << endl;

            
       //updating VMatrix
            if (arg  < -5.) 
                for (int i = 0; i <nDim; i++)
                {
                    for (int j=0; j<nDim; j++)
                    {
                        VMatrix(i,j) = 0;
                    } 
                } 
                
            else
                for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        VMatrix(i,j) = exp(arg);
                        sumv(i,j) += VMatrix(i,j);                                                
                        VMatrix(i,j) /= sumv(i,j);
                    }  
                }
                
                   for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        VMatrix(i,j) /= sumv(i,j);
                    }
                }   
                
            cout << "\n VMatrix is: \n" << VMatrix << endl;
    
    /********************************************************************/
    
      //Normalising rows of VMatrix
            Eigen::VectorXd sum1; 
            sum1 = VMatrix.rowwise().sum();
         //   cout << "\n sum1 is " << sum1(0) << endl;

              for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        VMatrix(i,j) = VMatrix(i,j)/sum1(j);
                    }
                }
    
    /*********************************************************************/
    
          //updating VMatrix again
            if ((arg = (min_Etest-Elocal)/kT) < -50. ) 
                for (int i = 0; i <nDim; i++)
                {
                    for (int j=0; j<nDim; j++)
                    {
                        VMatrix(i,j) = 0;
                    }
                }
            else
                for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        VMatrix(i,j) = exp(arg);
                        sumv(i,j) += VMatrix(i,j);
                    }
                }
              for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        VMatrix(i,j) /= sumv(i,j);
                    }
                }   
                
    /*********************************************************************/
         
         //Normalising columns of VMatrix
            Eigen::VectorXd sum2; 
            sum2 = VMatrix.colwise().sum();
           // cout << "\n sum2 is " << sum2(0) << endl;

              for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        VMatrix(i,j) = VMatrix(i,j)/sum2(i);
                    }
                }

          cout << "\n Normalised and Updated VMatrix is \n" << VMatrix << endl;

    /*********************************************************************/
      
        //Updating Propagator Matrix
            int Delta_ij;
            
           for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        if (i=j)
                            Delta_ij = 0;
                        else 
                            Delta_ij = 1;
                        sump(i,j) += PMatrix(i,j)*VMatrix(i,j);
                        PMatrix(i,j) = Delta_ij + VMatrix(i,j) + sump(i,j); 
                    }
                }
                cout << "\n PMatrix is:\n" << PMatrix << endl;
    /*********************************************************************/  
    
        //Computing L and R
        
                 
            vDeltaL = VMatrix.transpose() * DeltaMatrix;
            cout << "\n vDeltaL is: \n" << vDeltaL << endl;
            vdVecL = vDeltaL.diagonal();
            cout << "\n vdVecL is: \n" << vdVecL << endl;
            leftVec = PMatrix.transpose() * (TVec + vdVecL);
            cout << "\n leftVec is: \n" << leftVec << endl;

            vDeltaR = VMatrix * DeltaMatrix.transpose();
            cout << "\n vDeltaR is: \n" << vDeltaR << endl;
            vdVecR = vDeltaR.diagonal();
            cout << "\n vdVecR is: \n" << vdVecR << endl;
            rightVec = PMatrix * (TVec + vdVecR);
            cout << "\n rightVec is: \n" << rightVec << endl;
       
            valVec1 = leftVec.tail(nVehicles);
            cout << "\n valVec1 is: \n" << valVec1 << endl;
            valVec2 = rightVec.head(nVehicles);
            cout << "\n valVec2 is: \n" << valVec2 << endl;
            valVec << valVec1, valVec2;
            cout << "\n valVec is: \n" << valVec << endl;
            cout << "\n The maximum of the running times which should be minimized is: " << valVec.maxCoeff() << endl;
            
            //Updating the Energy function
            double maxleftVec, maxrightVec;
            maxleftVec = leftVec.maxCoeff();
            maxrightVec = rightVec.maxCoeff();
                      
            E_assign = 0.5*(maxleftVec + maxrightVec);
            cout << "E_assign is:" << E_assign << endl;
            Elocal = E_assign;
           
}

 ////////////////////////////////////////////////////////////////////////////////
 
    void NN_algo(Eigen::MatrixXd VMatrix, Eigen::MatrixXd PMatrix)
    {
         int FLAG = 1;
         while (FLAG)
		// update v, d and P
         {	update(VMatrix, PMatrix);
		kT *= kT_fac;
                cout << "\n kT is: " << kT << endl;
                cout << "\n Iteration done" << endl;
                             
        if (kT < kT_stop)
	FLAG = 0;
         }
        return;
    }
////////////////////////////////////////////////////////////////////////////////////

int main(int argc,char *argv[]) 
{
    clock_t tStart = clock();
    getVMatrix(nTasks,nVehicles); //Initialize the VMatrix
    cout << "\n VMatrix is: \n" << VMatrix << endl;
    Eigen::MatrixXd p = (I-VMatrix); //propagator matrix
    PMatrix = p.inverse();
    cout << "\n PMatrix is: \n" << PMatrix << endl; 
         
       ifstream file("TVec.txt");

       if (file.is_open())
       {
           for (int i=0; i<nDim; i++)
           {
               float item = 0.0;
               file >> item;
               TVec(i) = item;
           }
       }
       cout << "\n TVec is: \n" << TVec <<endl;
     
       ifstream file2("DeltaMatrix.txt");
       if (file2.is_open())
       {
            for (int i = 0; i < nDim; i++)
                for (int j = 0; j < nDim; j++)
            {
                float item2 = 0.0;
                file2 >> item2;
                DeltaMatrix(i,j) = item2;
            }
       } 
       
        cout << "\n DeltaMatrix is: \n" << DeltaMatrix <<endl;
       
       DeltaMatrix.diagonal().array() = 0;
       DeltaMatrix.leftCols(nVehicles) *= 0;
       DeltaMatrix.bottomRows(nVehicles) *= 0;
       DeltaMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  //adding all the constraints for the vehicles and tasks
       TauMatrix = (DeltaMatrix).colwise() + TVec;
       cout << "\n Updated DeltaMatrix is: \n" << DeltaMatrix << endl;
       cout << "\n TauMatrix is: \n" << TauMatrix << endl;
       
       NN_algo(VMatrix, PMatrix);    //NN_algo - updating equations
    //getSolutionStrings(vMatBest, TVec, deltaMat, nVehicles, nTasks)//Printing out the solutions
       printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    return 0;
}