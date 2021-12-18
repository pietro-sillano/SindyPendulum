#include<iostream>
#include <fstream>
#include<cmath>
#include<iomanip>


#define NX 51
#define NY 51


double Acceleration(double x); 
void   PositionVerlet(double &x,double &v,double (*Acceleration)(double),double dt);

using namespace std;


int main(){
  ofstream fdata;
  fdata.open("pendulum.dat");
  fdata  <<setiosflags(ios::scientific);
  fdata  <<setprecision(12);
  ofstream fdata2;
  fdata2.open("gaussian.dat");
  fdata2 <<setiosflags(ios::scientific);
  fdata2 <<setprecision(12);

  double t;
  double z[NX][NY];

  double x[NX];
  double y[NY];
  int i,j;
  
  
  double xa = -1.5 ;
  double xb = 1.5  ;
  double ya = -1.5 ;
  double yb = 1.5  ;
  x[0] = xa;
  y[0] = ya;
  
  double dx=(xb-xa)/double(NX-1);
  double dy=(yb-ya)/double(NY-1);

  for(i=0;i<NX;i++){
    x[i]=x[0]+i*dx;
  }
  
  for(i=0;i<NY;i++){
    y[i]=y[0]+i*dy;
  }
  
  //   CONDIZIONI INIZIALI
  double theta0 = 0.5;
  double w0=0.;
  
  double theta=theta0;
  double w=w0;
  
//   CONDIONI TEMPORALI
  t=0;
  double tmax = 5.;
  double dt=0.1;
  
  while (t < tmax ){
    PositionVerlet(theta,w,Acceleration,dt);
    fdata  <<t<<"  " <<theta<<"  "<<w<<"  "<<endl;
    for(i=0;i<NX;i++){
      for(j=0;j<NY;j++){
        z[i][j] = exp(- 20*((x[i] - cos(theta -M_PI/2))*(x[i] - cos(theta -M_PI/2))) - 20 * ((y[j] -sin(theta-M_PI/2))*(y[j] -sin(theta-M_PI/2))));
        fdata2 <<x[i]<<"  "<<y[j]<<"  "<<z[i][j]<<endl;
    }
      fdata2 <<"\n";
    }
    
    t+=dt;
    fdata2 <<"\n";
    fdata2 <<"\n";

  }
  
  cout<<"VERLET FINISHED" <<endl;
  fdata.close();
  fdata2.close();

  return 0;
}


double Acceleration(double x){
  return -sin(x);
}
void PositionVerlet(double &x,double &v,double (*Acceleration)(double),double dt){
  double acc;
  x+=dt*0.5*v;
  acc=Acceleration(x);
  v+=dt*acc;
  x+=0.5*dt*v;
}
