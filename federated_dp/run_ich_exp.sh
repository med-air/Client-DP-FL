# no dp
# DP-FedAvg
python fed_train.py --mode fedavg --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --seed 0 &&
python fed_train.py --mode fedavg --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --seed 1 &&
python fed_train.py --mode fedavg --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --seed 2
# DP-FedAdam
python fed_train.py --mode fedadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --seed 0 &&
python fed_train.py --mode fedadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --seed 1 &&
python fed_train.py --mode fedadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --seed 2
# DP-FedNova
python fed_train.py --mode fednova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --seed 0 &&
python fed_train.py --mode fednova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --seed 1 &&
python fed_train.py --mode fednova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --seed 2
# DP2-RMSProp
python fed_train.py --mode fed2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --seed 0 &&
python fed_train.py --mode fed2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --seed 1 &&
python fed_train.py --mode fed2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --seed 2


# z = 0.5 
# DP-FedAvg
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 0 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 1 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 2
# DP-FedAdam
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 0 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 1 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 0 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 1 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 2
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 0 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 1 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 0.5 --seed 2


# z=0.5  + our method
# DP-FedAvg
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 0 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 1 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 2
# DP-FedAdam
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 0 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 1 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 0 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 1 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 2
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 0 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 1 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 0.5 --ada_vn --seed 2



# z = 1.0
# DP-FedAvg
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 0 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 1 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 2
# DP-FedAdam
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 0 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 1 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 0 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 1 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 2
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 0 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 1 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.0 --seed 2


# z=1.0  + our method
# DP-FedAvg
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 0 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 1 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 2
# DP-FedAdam
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 0 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 1 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 0 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 1 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 2
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 0 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 1 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.0 --ada_vn --seed 2



# z = 1.5
# DP-FedAvg
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 0 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 1 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 2
# DP-FedAdam
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 0 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 1 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 0 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 1 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 2
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 0 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 1 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.5 --seed 2


# z=1.5  + our method
# DP-FedAvg
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 0 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 1 &&
python fed_train.py --mode dpsgd --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 2
# DP-FedAdam
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 0 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 1 &&
python fed_train.py --mode dpadam --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adam_lr 0.003 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 2
# DP-FedNova
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 0 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 1 &&
python fed_train.py --mode dpnova --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 2
# DP2-RMSProp
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 0 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 1 &&
python fed_train.py --mode dp2rmsprop --data RSNA-ICH --batch 16 -N 20 -VN 1 --rounds 100 --lr 3e-4 --rmsprop_lr 3e-5 --adaclip --delta 0.01 --noise_multiplier 1.5 --ada_vn --seed 2