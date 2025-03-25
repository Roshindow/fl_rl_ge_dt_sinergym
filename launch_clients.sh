#!/bin/bash
for i in {1..20}
do
	gnome-terminal --tab -- bash -c "sudo -u $USER docker run -v fl_exp:/c$i\_data -it --rm --network host sinergym-flower-client --environment_name Eplus-5zone-mixed-discrete-stochastic-v1 --jobs 1 --n_actions 10 --learning_rate 0.001 --df 0.05 --input_space 11 --episodes 10 --episode_len 822 --lambda_ 10 --generations 5 --cxp 0.2 --mp 1 --low -1 --up 1 --types '#1,13,1,1;0,24,1,1;-10,50,1,1;0,100,1,1;0,300,1,1;0,1000,1,1;-10,50,1,1;-10,50,1,1;-10,50,1,1;0,100,1,1;0,50,1,1' --mutation 'function-tools.mutUniformInt#low-0#up-40000#indpb-0.1' --patience 30 --timeout 600 --grammar_angle 1 --seed $i --clients=1; exec bash"
done
