# fl_rl_ge_dt_sinergym
Code repository for Enrico Micheli's 2024 Master thesis

## Usage

### Server
Build the server using the command:
```
sudo docker build -f Dockerfile_server -t custom-flower-server .
```

Launch the server using a command such as:
```
sudo docker run -it --name flower-server --rm --network host -p 8080:8080 custom-flower-server --address flower-server:8080 --rounds 20 --min_fit_clients 1 --min_eval_clients 1 --min_available_clients 1
```

### Client
Build the client using the command:
```
sudo docker build -f Dockerfile_client -t sinergym-flower-client .
```

Launch the client using a command such as:
```
sudo docker run -v fl_exp:/c1_data -it --rm --network host sinergym-flower-client --environment_name Eplus-5zone-mixed-discrete-stochastic-v1 --jobs 10 --n_actions 10 --learning_rate 0.001 --df 0.05 --input_space 11 --episodes 10 --episode_len 822 --lambda_ 10 --generations 5 --cxp 0.2 --mp 1 --low -1 --up 1 --types '#1,13,1,1;0,24,1,1;-10,50,1,1;0,100,1,1;0,300,1,1;0,1000,1,1;-10,50,1,1;-10,50,1,1;-10,50,1,1;0,100,1,1;0,50,1,1' --mutation 'function-tools.mutUniformInt#low-0#up-40000#indpb-0.1' --patience 30 --timeout 600 --grammar_angle 1 --seed 1 --clients=1
```

Alternatively, launch_clients.sh launches n clients according to the command it contains. **Use this or manual multiple launches rather than the clients parameter, some problems occur due to gRPC messaging otherwise**
