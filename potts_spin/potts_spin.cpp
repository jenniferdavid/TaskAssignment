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
double kT_start = 1000;
double kT_stop  = 0.001;
double kT_swfac = 0.095;
double kT_fac = exp( log(kT_swfac) / (nVehicles * nDim) ); // lower T after every neuron update
double kT = kT_start * kT_fac;  // * to give the ini conf a own datapoint in sat plot
double Pji_coeff = 0.001;
double sumv;
static const double small = 1e-15;
static const double onemsmall = 1 - small;
static const double lk0 = 1/small - 1; 
double lk;
  
//////////////////////////////////////////////////

Eigen::MatrixXd VMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd UpdatedVMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd DeltaVMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd DeltaPMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd PMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd dQ = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd P = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd PP = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd PPP = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd UpdatedPMatrix = MatrixXd::Zero(nDim,nDim);

Eigen::MatrixXd DeltaMatrix = MatrixXd::Ones(nDim,nDim);
Eigen::MatrixXd TauMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd I = MatrixXd::Identity(nDim,nDim); //identity matrix

Eigen::MatrixXd E_local = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd E_loop = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd E_assign = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd E = MatrixXd::Zero(nDim,nDim);

Eigen::MatrixXd initialE_local = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd initialE_loop = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd initialE_assign = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd initialE = MatrixXd::Zero(nDim,nDim);

Eigen::VectorXd TVec(nDim);

Eigen::VectorXd vdVecL(nDim);
Eigen::VectorXd vdVecR(nDim);
Eigen::VectorXd rightVec(nDim);
Eigen::VectorXd leftVec(nDim);
Eigen::MatrixXd vDeltaR;
Eigen::MatrixXd vDeltaL;

Eigen::VectorXd ivdVecL(nDim);
Eigen::VectorXd ivdVecR(nDim);
Eigen::VectorXd irightVec(nDim);
Eigen::VectorXd ileftVec(nDim);
Eigen::MatrixXd ivDeltaR;
Eigen::MatrixXd ivDeltaL;

/////////////////////////////////////////////////////////

class neural
{

  public:
  
  void getVMatrix ()
  {
        cout << "\n No. of Vehicles: " << nVehicles << endl;
        cout << "\n No. of Tasks: " << nTasks << endl;
        cout << "\n Dimension of VMatrix is: " << nDim << endl;
        double r = ((double) rand() / (RAND_MAX))/80;
        cout << "\n r is: " << r << endl;
        double tmp = 1./(rDim-1);
        cout << "\n tmp is: " << tmp << endl;

        for (int i = 0; i < nDim; i++)
        {
            for (int j = 0; j < nDim; j++)
            {
                VMatrix(i,j) = tmp;// + (((double) rand() / (RAND_MAX))/80) ;// + 0.02*(rand() - 0.5);	// +-1 % noise
            }
        }
      //  cout << "\n VMatrix is: \n" << VMatrix << endl;

        VMatrix.diagonal().array() = 0;
        VMatrix.leftCols(nVehicles) *= 0;
        VMatrix.bottomRows(nVehicles) *= 0;
        VMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  //adding all the constraints for the vehicles and tasks
        cout << "\n VMatrix is: \n" << VMatrix << endl;
     
        PMatrix = I + (VMatrix) + (VMatrix)*(VMatrix) + (VMatrix)*(VMatrix)*(VMatrix) + (VMatrix)*(VMatrix)*(VMatrix)*(VMatrix);
        cout << "\n PMatrix is as in (I + V^2 + V^3+ ...): \n" << PMatrix << endl; 
        
        P = (I - VMatrix).inverse();
        cout << "\n PMatrix is as in (I - V)^(-1): \n" << P << endl; 

        cout << "\n ///////////////////////////////////////////// " << endl;
     
  }
    
  ///////////////////////////////////////////////////////////////////////
  void NN_algo() const
  {    
         int FLAG = 1;
         
            ivDeltaL = VMatrix.transpose() * DeltaMatrix;
            ivdVecL = ivDeltaL.diagonal();
            ileftVec = PMatrix.transpose() * (TVec + ivdVecL);

            ivDeltaR = VMatrix * DeltaMatrix.transpose();
            ivdVecR = ivDeltaR.diagonal();
            irightVec = PMatrix * (TVec + ivdVecR);
            
            MatrixXf::Index imaxl, imaxr;
            
            float imaxleftVecInd, imaxrightVecInd, iX, iY;
            imaxleftVecInd = ileftVec.maxCoeff(&imaxl);
            imaxrightVecInd = irightVec.maxCoeff(&imaxr);
         
          for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        iX = (ileftVec(i) + DeltaMatrix(i,j)) * PMatrix(j,imaxl);
                        iY = (irightVec(i) + DeltaMatrix(i,j)) * PMatrix(imaxr,i);
                        initialE_assign(i,j) = 0.5*(iX + iY);
                        
                        double initial_lk = PMatrix(j,i) / PMatrix(i,i);	// the "zeroed" Pji
                            if ( initial_lk < onemsmall )
                                initial_lk = initial_lk/(1-initial_lk); // => the resulting Pji for choice j
                            else
                                initial_lk = lk0;  
                        initialE_loop(i,j) = initial_lk;
                        
                        initialE_local(i,j) = initialE_assign(i,j) + initialE_loop(i,j);
                        
                    }
                }
         
         cout << "\n initial Elocal is: \n" << initialE_local << endl;
         E_local = initialE_local;
       
         cout << "\n ///////////////////////////////////////////// " << endl;
                
         cout << "\n Initial VMatrix is: \n" << VMatrix << endl;
         cout << "\n Initial PMatrix is: \n" << PMatrix << endl;
         cout << "\n TauMatrix is: \n" << TauMatrix << endl;
         cout << "\n Elocal is: \n" << E_local << endl;
        
         cout << "\n ///////////////////////////////////////////// " << endl;
        
         int iteration = 1;
         
         while (FLAG)
         // update v, d and P
         {	
            cout << "\n" << iteration << " ITERATION STARTING" << endl;
            cout << "\n kT is " << kT << endl;
            cout << "\n VMatrix before is \n" << VMatrix << endl;
            cout << "\n UpdatedVMatrix before is \n" << UpdatedVMatrix << endl;
            cout << "\n PMatrix before is \n" << PMatrix << endl;
            cout << "\n UpdatedPMatrix before is \n" << UpdatedPMatrix << endl;

            //updating VMatrix
           
                        E = (E_local/kT);
                   
            for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        
                        if (VMatrix(i,j) == 0)
                            { UpdatedVMatrix(i,j) = 0;}
                        
                        else 
                            { UpdatedVMatrix(i,j) = std::exp (E(i,j));}
                    }  
                }

            cout << "\n Updating VMatrix is: (my ref) \n" << UpdatedVMatrix << endl;
            
            Eigen::VectorXd sumv; 
            sumv = UpdatedVMatrix.rowwise().sum();
            cout << "sumv is (my ref): \n " << sumv << endl;
            
             for (int i = 0; i < nDim; i++)
                {                    
                    for (int j=0; j< nDim; j++)
                    {
                        if (VMatrix(i,j) == 0)
                            UpdatedVMatrix(i,j) = 0;
                        else
                        UpdatedVMatrix(i,j) /= sumv(i);
                    }  
                }   
           cout << "\n After Updation, UpdatedVMatrix is: \n" << UpdatedVMatrix << endl;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Normalising till the values along the row/columns is zero
            cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
            cout << "\n NORMALISATION BEGINS \n" << endl;

            int x = 0;
           LABEL:
            Eigen::VectorXd sum_row; 
            sum_row = UpdatedVMatrix.rowwise().sum();
            cout << "\n row sum before row normalisation is \n" << sum_row << endl;
            
             for (int k = 0; k < sum_row.size(); k++)
                {
                    if (sum_row(k) == 0.000)
                        {
                            cout << "\n Row " << k << " is with constraints, so skipping" << endl;
                        }
                    else if (sum_row(k) == 1.000)
                        {
                            cout << "\n Row " << k << " is already normalised" << endl;
                        }
                    else 
                        {
                            cout << "\n Row Normalising " << endl;
                            for (int i = 0; i < nDim; i++)
                                {
                                    UpdatedVMatrix(k,i) = UpdatedVMatrix(k,i)/sum_row(k);
                                }
                        }
                }                  
            cout << "\n So finally, the Row Normalised UpdatedVMatrix is \n" << UpdatedVMatrix << endl;
            cout << "\n row sum after row normalisation is \n" << UpdatedVMatrix.rowwise().sum() << endl;
   
            //Normalising columns of VMatrix
            Eigen::VectorXd sum_col; 
            sum_col = UpdatedVMatrix.colwise().sum();
            cout << "\n col sum before column normalisation is \n" << sum_col.transpose() << endl;

            for (int k = 0; k < sum_col.size(); k++)
                {
                    if (sum_col(k) == 0)
                        {
                            cout << "\n Col " << k << " is with constraints, so skipping" << endl;
                        }
                    else if (sum_col(k) == 1)
                        {
                            cout << "\n Col " << k << " is already normalised" << endl;
                        }   
                    else 
                        {
                            cout << "\n Column Normalising \n" << endl;
                            for (int j = 0; j < nDim; j++)
                                {
                                    UpdatedVMatrix(j,k) = UpdatedVMatrix(j,k)/sum_col(k);
                                }
                        }
                }
            cout << "\n Col Normalised UpdatedVMatrix is \n" << UpdatedVMatrix << endl;
            cout << "\n col sum after column normalisation is \n" << UpdatedVMatrix.colwise().sum() << endl;
            cout << "\n row sum after column normalisation is \n" << UpdatedVMatrix.rowwise().sum() << endl;
            x++;
            
            if (x != 7)
                goto LABEL;
            else
            {
                cout << "\n NORMALISATION END \n" << endl;
                cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
                cout << "\n Final row and column normalised UpdatedVMatrix is \n" << UpdatedVMatrix << endl;
                cout << "\n  Previous VMatrix is \n" << VMatrix << endl;
                
                DeltaVMatrix = (UpdatedVMatrix - VMatrix);
                cout << "\n Delta V Matrix is \n" << DeltaVMatrix << endl;
                
                cout << "\n PMatrix old is \n" << PMatrix << endl;             
                dQ = DeltaVMatrix * PMatrix;    
            
                cout << "\n dQ:\n" << dQ << endl;
                cout << "\n trace of dQ:\n" << dQ.trace() << endl;
                cout << "\n 1 - (trace of dQ) :\n" << (1 - dQ.trace()) << endl;

                DeltaPMatrix = (PMatrix * dQ)/(1-(dQ.trace()));
                cout << "\n Delta PMatrix:\n" << DeltaPMatrix << endl;
             
                UpdatedPMatrix = PMatrix + DeltaPMatrix;
                cout << "\n Updated PMatrix using absolute method is:\n" << UpdatedPMatrix << endl;
                
                PP = (I-UpdatedVMatrix);
                PPP = PP.inverse();
                cout << "\n Updated PMatrix using exact inverse is:\n" << PPP << endl;
                
                //Computing L and R
                vDeltaL = UpdatedVMatrix.transpose() * DeltaMatrix;
                cout << "\n vDeltaL is: \n" << vDeltaL << endl;
                vdVecL = vDeltaL.diagonal();
                cout << "\n vdVecL is: \n" << vdVecL << endl;
                leftVec = UpdatedPMatrix.transpose() * (TVec + vdVecL);
                cout << "\n leftVec is: \n" << leftVec << endl;

                vDeltaR = UpdatedVMatrix * DeltaMatrix.transpose();
                cout << "\n vDeltaR is: \n" << vDeltaR << endl;
                vdVecR = vDeltaR.diagonal();
                cout << "\n vdVecR is: \n" << vdVecR << endl;
                rightVec = UpdatedPMatrix * (TVec + vdVecR);
                cout << "\n rightVec is: \n" << rightVec << endl;
       
                MatrixXf::Index maxl, maxr;
                float maxleftVecInd = leftVec.maxCoeff(&maxl);
                float maxrightVecInd = rightVec.maxCoeff(&maxr);
            
                double X,Y;
                //Updating the Energy function
                for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        X = ((leftVec(i) + DeltaMatrix(i,j)) * UpdatedPMatrix(j,maxl));
                        Y = ((rightVec(i) + DeltaMatrix(i,j)) * UpdatedPMatrix(maxr,i));
                        E_assign(i,j) = 0.5*(X + Y);
                        
                        double lk = UpdatedPMatrix(j,i) / UpdatedPMatrix(i,i);	
                            if ( lk < onemsmall )
                                lk = lk/(1-lk); 
                            else
                                lk = lk0;  
                        E_loop(i,j) = lk;
                        E_local(i,j) = E_loop(i,j) + E_assign(i,j);
                    }
                }
                cout << "\n E_local is: \n" << E_local << endl;
  
                kT *= kT_fac;
                cout << "\n new kT is: " << kT << endl;
                iteration ++;
                cout << "\n" << iteration << " ITERATION DONE" << endl;
                VMatrix = UpdatedVMatrix;
                PMatrix = UpdatedPMatrix;
                cout << "\n /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/ " << endl;

                if (kT < kT_stop) 
                FLAG = 0;
            }
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
       }
       
    else 
       cout << "file not open" << endl;
    
    cout << "\n TVec is: \n" << TVec <<endl;
     
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
       
    nn.NN_algo();    //NN_algo - updating equations
    cout << "\n Annealing done \n" << endl;
    //nn.displaySolution();//Printing out the solutions
    printf("Total time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}
