plot './series/epochLoss.dat' u 1:2 w lp lw 1.5 lc 8 title "Loss Plot with Training"
set xlabel "Epochs"
set ylabel "Model Loss"
set grid