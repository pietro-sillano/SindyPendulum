SESSION=1


if (SESSION==1){

plot 'pendulum.dat' u 1:2 w l
pause -1

}


if (SESSION==2){
reset

# set terminal pngcairo size 350,262 
# system('mkdir -p png')


set terminal gif animate delay 10
set output 'gaussian.gif'
set border 0
do for [i=0:1000] {
  set pm3d map
  set pm3d interpolate 1,1
  set title  sprintf('Iter: %d',i) font ",18"
  splot 'gaussian.dat' using 1:2:3 index i
}
}





