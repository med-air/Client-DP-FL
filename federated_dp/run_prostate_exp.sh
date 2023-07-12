# no dp
# DP-FedAvg
python fed_train.py --mode fedavg --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --seed 0 &&
python fed_train.py --mode fedavg --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --seed 1 &&
python fed_train.py --mode fedavg --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --seed 2
# DP-FedAdam
python fed_train.py --mode fedadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --seed 0 &&
python fed_train.py --mode fedadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --seed 1 &&
python fed_train.py --mode fedadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --seed 2
# DP-FedNova
python fed_train.py --mode fednova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --seed 0 &&
python fed_train.py --mode fednova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --seed 1 &&
python fed_train.py --mode fednova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --seed 2
# DP2-RMSProp
python fed_train.py --mode fed2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-3 --seed 0 &&
python fed_train.py --mode fed2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-3 --seed 1 &&
python fed_train.py --mode fed2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-3 --seed 2


# z = 0.3
# DP-FedAvg
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 2 
# DP-FedAdam
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 2 
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.3 --adaclip --delta 0.1 --seed 2

# z = 0.3 + our method
# DP-FedAvg
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 2 
# DP-FedAdam
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 2 
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.3 --adaclip --delta 0.1 --ada_vn --seed 2



# z = 0.5
# DP-FedAvg
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 2
# DP-FedAdam
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 2 
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.5 --adaclip --delta 0.1 --seed 2


# z = 0.5  + our method
# DP-FedAvg
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 2 
# DP-FedAdam
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 2 
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.5 --adaclip --delta 0.1 --ada_vn --seed 2



# z = 0.7   
# DP-FedAvg
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 2
# DP-FedAdam
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 2 
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 0 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 1 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.7 --adaclip --delta 0.1 --seed 2


# z = 0.7  + our method
# DP-FedAvg
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dpsgd --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 2 
# DP-FedAdam
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dpadam --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --adam_lr 0.03 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dpnova --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 2 
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 0 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 1 &&
python fed_train.py --mode dp2rmsprop --data prostate --batch 8 -N 6 -VN 1 --rounds 100 --lr 0.001 --rmsprop_lr 1e-4 --noise_multiplier=0.7 --adaclip --delta 0.1 --ada_vn --seed 2