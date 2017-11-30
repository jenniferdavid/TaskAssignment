/* Trials with Potts_Spin based optimization.
 *
 * Copyright (C) 2014 Jennifer David. All rights reserved.
 *
 * BSD license:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of
 *    contributors to this software may be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR THE CONTRIBUTORS TO THIS SOFTWARE BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
   \file potts_spin.cpp
   
   Trials with running neural network based optimization method for task assignment. 
   
*/
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

using namespace std;
using namespace Eigen;


//////////////////////////////////////////////////

int nVehicles = 1;
int nTasks = 3;
int nDim = 2*nVehicles + nTasks;
int rDim = nTasks + nVehicles;
double kT_start = 10.0;
double kT_stop  = 0.001;
double kT_swfac = 0.95;
double kT_fac = exp( log(kT_swfac) / (nVehicles * nDim) ); // lower T after every neuron update
double kT = kT_start * kT_fac;  // * to give the ini conf a own datapoint in sat plot
double initial_Elocal, initial_Eloop, initial_Eassign;
double Pji_coeff = 0.001;
static const double small = 1e-15;
static const double onemsmall = 1 - small;
static const double lk0 = 1/small - 1; 
double lk;
  
//////////////////////////////////////////////////

Eigen::MatrixXd Elocal = MatrixXd::Zero(rDim,rDim);
Eigen::MatrixXd Eassign = MatrixXd::Zero(rDim,rDim);
Eigen::MatrixXd Eloop = MatrixXd::Zero(rDim,rDim);

Eigen::MatrixXd Energy_Matrix = MatrixXd::Zero(rDim,rDim);
Eigen::MatrixXd VMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd Core_Matrix = MatrixXd::Zero(rDim,rDim);
Eigen::MatrixXd PCore_Matrix = MatrixXd::Zero(rDim,rDim);
Eigen::MatrixXd Delta_Core_Matrix = MatrixXd::Zero(rDim,rDim);
Eigen::MatrixXd Tau_Core_Matrix = MatrixXd::Zero(rDim,rDim);
Eigen::MatrixXd DeltaMatrix = MatrixXd::Ones(nDim,nDim);
Eigen::MatrixXd TauMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd vDeltaR;
Eigen::MatrixXd vDeltaL;
Eigen::MatrixXd sump = MatrixXd::Zero(rDim,rDim);
Eigen::MatrixXd I = MatrixXd::Identity(rDim,rDim); //identity matrix

Eigen::VectorXd TVec(nDim);
Eigen::VectorXd TRVec(rDim);
Eigen::VectorXd vdVecL(nDim);
Eigen::VectorXd vdVecR(nDim);
Eigen::VectorXd rightVec(nDim);
Eigen::VectorXd leftVec(nDim);
Eigen::VectorXd valVec1(nVehicles);
Eigen::VectorXd valVec2(nVehicles);
Eigen::VectorXd valVec(nVehicles+nVehicles);

/////////////////////////////////////////////////////////

class neural
{

  public:
  
  void getVMatrix ()
  {
      
        cout << "\n No. of Vehicles: " << nVehicles << endl;
        cout << "\n No. of Tasks: " << nTasks << endl;
        cout << "\n Dimension of VMatrix is: " << nDim << endl;
        double r = ((double) rand() / (RAND_MAX));
        cout << "\n r is: " << r << endl;
        double tmp = 1./nDim;
        cout << "\n tmp is: " << tmp << endl;

        for (int i = 0; i < nDim; i++)
        {
            for (int j = 0; j < nDim; j++)
            {
                VMatrix(i,j) = tmp + (((double) rand() / (RAND_MAX))/2) ;// + 0.02*(rand() - 0.5);	// +-1 % noise
            }
        }
        cout << "\n VMatrix is: \n" << VMatrix << endl;

        VMatrix.diagonal().array() = 0;
        VMatrix.leftCols(nVehicles) *= 0;
        VMatrix.bottomRows(nVehicles) *= 0;
        VMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  //adding all the constraints for the vehicles and tasks
        cout << "\n VMatrix is: \n" << VMatrix << endl;
     
        Core_Matrix = VMatrix.block(0, nVehicles, rDim, rDim);
        double tmp2 = 1./(rDim-1);
        cout << "\n tmp2 is: " << tmp2 << endl;
        
        for (int i = 0; i < rDim; i++)
        {
            for (int j = 0; j < rDim; j++)
            {
                if (Core_Matrix(i,j) == 0)
                    Core_Matrix(i,j) = 0;
                else Core_Matrix(i,j) = tmp2 + (((double) rand() / (RAND_MAX))/2);
            }
        }
        cout << "\n Core VMatrix is: \n" << Core_Matrix << endl;
        Eigen::MatrixXd p = (I-Core_Matrix); //propagator matrix
        
        cout << "\n I-VMatriz is: \n" << p<< endl;
        PCore_Matrix = p.inverse();
        cout << "\n PCore Matrix is: \n" << PCore_Matrix << endl; 
        
  }
  //////////////////////////////////////////////////////////////////////
  void initial_energy() const
  {
     // initial_Eloop = 
     // initial_Eassign = 
     // initial_Elocal = initial_Eassign + initial_Eloop;
      
      initial_Eassign = initial_Eloop;
      
  }
  ///////////////////////////////////////////////////////////////////////
  void NN_algo() 
  {    
         int FLAG = 1;
         double value;
         cout << "\n ///////////////////////////////////////////// " << endl;
         
         cout << "\n Initial VMatrix is: \n" << Core_Matrix << endl;
         cout << "\n Initial PMatrix is: \n" << PCore_Matrix << endl;
         cout << "\n TauMatrix is: \n" << Tau_Core_Matrix << endl;
         cout << "\n Elocal is: " << initial_Elocal << endl;
        
         cout << "\n ///////////////////////////////////////////// " << endl;

         while (FLAG)
         // update v, d and P
         {	
            cout << "\n kT is " << kT << endl;
            cout << "\n Core VMatrix is \n" << Core_Matrix << endl;
    
                for (int i = 0; i < rDim; i++)
                {
                    for (int j=0; j< rDim; j++)
                    {
                       // Eassign(i,j) = 0.5 * ();
                        Elocal(i,j) = Eassign(i,j);
                    }  
                }
                            
            //updating VMatrix
            for (int i = 0; i < rDim; i++)
                {
                    for (int j=0; j< rDim; j++)
                    {
                        if (Core_Matrix(i,j) == 0)
                            Core_Matrix(i,j) = 0;
                        else
                            Energy_Matrix(i,j) = Elocal(i,j);
                            value = Energy_Matrix(i,j) / kT;
                            Core_Matrix(i,j) = exp(value);
                    }  
                }
                
            Eigen::VectorXd sumv;         
            sumv = Core_Matrix.rowwise().sum();
            for (int i = 0; i < rDim; i++)
                {                    
                    for (int j=0; j< rDim; j++)
                    {
                        Core_Matrix(i,j) /= sumv(i);
                    }  
                }   
                
            cout << "\n Updated Core VMatrix is: \n" << Core_Matrix << endl;
            
//////////////////////////////////////////////////////////////////////////////

            //Normalising rows of VMatrix
            Eigen::VectorXd sum1; 
            sum1 = Core_Matrix.rowwise().sum();
            cout << "\n row sum is " << sum1 << endl;

            for (int i = 0; i < rDim; i++)
                {
                    for (int j=0; j< rDim; j++)
                    {
                        Core_Matrix(i,j) = Core_Matrix(i,j)/sum1(j);
                    }
                }
    
            cout << "\n Row Normalised Core VMatrix is \n" << Core_Matrix << endl;

            
///////////////////////////////////////////////////////////////////////         
    
            //Normalising columns of VMatrix
            Eigen::VectorXd sum2; 
            sum2 = Core_Matrix.colwise().sum();
            cout << "\n col sum is " << sum2 << endl;

            for (int i = 0; i < rDim; i++)
                {
                    for (int j=0; j< rDim; j++)
                    {
                        Core_Matrix(i,j) = Core_Matrix(i,j)/sum2(i);
                    }
                }
            cout << "\n Col Normalised Core VMatrix is \n" << Core_Matrix << endl;
    
/////////////////////////////////////////////////////////////////////////

            double Delta_im;
            for (int i = 0; i < rDim; i++)
                {
                    for (int m=0; m< rDim; m++)
                    {                      
                        for (int j = 0; j < rDim; j++)
                                {
                                    if (i==j)
                                        Delta_im = 0;
                                    else 
                                        Delta_im = 1;
                                        Delta_im += Core_Matrix(i,j)*PCore_Matrix(j,m);
                                        PCore_Matrix(i,j) = Delta_im;
                                }
                    }
                }
            cout << "\n Updated PCore Matrix is:\n" << PCore_Matrix << endl;
           
/////////////////////////////////////////////////////////////////////////    
       
            //Computing L and R
            vDeltaL = Core_Matrix.transpose() * Delta_Core_Matrix;
            cout << "\n vDeltaL is: \n" << vDeltaL << endl;
            vdVecL = vDeltaL.diagonal();
            cout << "\n vdVecL is: \n" << vdVecL << endl;
            leftVec = PCore_Matrix.transpose() * (TRVec + vdVecL);
            cout << "\n leftVec is: \n" << leftVec << endl;

            vDeltaR = Core_Matrix * Delta_Core_Matrix.transpose();
            cout << "\n vDeltaR is: \n" << vDeltaR << endl;
            vdVecR = vDeltaR.diagonal();
            cout << "\n vdVecR is: \n" << vdVecR << endl;
            rightVec = PCore_Matrix * (TRVec + vdVecR);
            cout << "\n rightVec is: \n" << rightVec << endl;
       
            valVec1 = leftVec.tail(nVehicles);
            cout << "\n valVec1 is: \n" << valVec1 << endl;
            valVec2 = rightVec.head(nVehicles);
            cout << "\n valVec2 is: \n" << valVec2 << endl;
            valVec << valVec1, valVec2;
            cout << "\n valVec is: \n" << valVec << endl;
            cout << "\n The maximum of the running times which should be minimized is: " << valVec.maxCoeff() << endl;
            
            //Computing derviatives of L and R
            
            //Computing the Energy function
            double maxleftVec, maxrightVec;
            maxleftVec = leftVec.maxCoeff();
            maxrightVec = rightVec.maxCoeff();
            
            for (int i = 0; i < rDim; i++)
                {
                    for (int j=0; j< rDim; j++)
                    {
                        Eassign(i,j) = 0.5 * ();
                    }
                }
            
           // Eassign = 0.5*(maxleftVec + maxrightVec);
            
            cout << "Eassign is:" << Eassign << endl;
            
            Elocal = Eassign;
            cout << "Elocal is:" << Elocal << endl;

            kT *= kT_fac;
                cout << "\n new kT is: " << kT << endl;
                cout << "\n Iteration done" << endl;
                             
            if (kT < 9)
            FLAG = 0;
            }
  }
};

/////////////////////////////////////////////////////////////////////////

int main(int argc,char *argv[])
{    
    neural nn;
    clock_t tStart = clock();
    nn.getVMatrix(); //Initialize the VMatrix
   
    ifstream file("TVec.txt");
    if (file.is_open())
       {
           for (int i=0; i<nDim; i++)
           {
               float item;
               file >> item;
               TVec(i) = item;
           }
           
           for (int i=0; i<(nDim-nVehicles); i++)
           {
               float item1;
               file >> item1;
               TRVec(i) = item1;
           }
       }
       
    else 
       cout << "file not open" << endl;
    
    cout << "\n TVec is: \n" << TVec <<endl;
    cout << "\n TRVec is: \n" << TRVec <<endl;
     
    ifstream file2("DeltaMatrix.txt");
    if (file2.is_open())
       {
            for (int i = 0; i < nDim; i++)
                for (int j = 0; j < nDim; j++)
                    {
                        float item2;
                        file2 >> item2;
                        DeltaMatrix(i,j) = item2;
                    }
       } 
    else
       cout <<"file not open"<<endl;
   
    cout << "\n DeltaMatrix is: \n" << DeltaMatrix <<endl;
    DeltaMatrix.diagonal().array() = 0;
    DeltaMatrix.leftCols(nVehicles) *= 0;
    DeltaMatrix.bottomRows(nVehicles) *= 0;
    DeltaMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  //adding all the constraints for the vehicles and tasks
       
    TauMatrix = (DeltaMatrix).colwise() + TVec;
    cout << "\n Updated DeltaMatrix is: \n" << DeltaMatrix << endl;
    cout << "\n TauMatrix is: \n" << TauMatrix << endl;
       
    Delta_Core_Matrix = DeltaMatrix.block(0, nVehicles, rDim, rDim);
    Tau_Core_Matrix = TauMatrix.block(0, nVehicles, rDim, rDim);
    cout << "\n Reduced DeltaMatrix is: \n" << Delta_Core_Matrix << endl;
    cout << "\n Reduced TauMatrix is: \n" << Tau_Core_Matrix << endl;
    
    nn.initial_energy(); //initial update
    nn.NN_algo();    //NN_algo - updating equations
    cout << "\n Annealing done \n" << endl;
   // nn.displaySolution();//Printing out the solutions
    printf("Total time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}
