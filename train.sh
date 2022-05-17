export CUBLAS_WORKSPACE_CONFIG=:16:8

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

DATE=`date +%Y-%m-%d`
echo $DATE
DIRECTORY=./save/${DATE}/
if [ ! -d "./save" ]; then
    mkdir ./save
fi

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configuration   ##############################

epoch=200
STEP=100
RANDOM_SEED=10

DATA_ROOT='./dataset' #YOUR DATASET ROOT
if [ ! -d "$DATA_ROOT" ]; then
    mkdir $DATA_ROOT
fi
DATASET='cifar10'
# DATASET='svhn'
# DATASET='gtsrb'
if [ ! -d "./$DATA_ROOT/$DATASET" ]; then
    mkdir ./$DATA_ROOT/$DATASET
fi


############### Train   ##############################
# ----- IP vendor: Train biased models -----
echo "train model for IP vendor"

# train models

MODEL='MobileNet'
# MODEL='resnet18'
# MODEL='ShuffleNetG2'

# log path
save_path=save/${DATE}/${DATASET}_${MODEL}

# To train model A
python train_classifier.py --dataset ${DATASET} \
                            --model ${MODEL} \
                            --n_epochs ${epoch} \
                            --data_root ${DATA_ROOT} \
                            --manualSeed ${RANDOM_SEED} \
                            --save_path ${save_path} \
                            --class_weight 0 

wait 

# To train model B
python train_classifier.py --dataset ${DATASET} \
                            --model ${MODEL} \
                            --n_epochs ${epoch} \
                            --data_root ${DATA_ROOT} \
                            --manualSeed ${RANDOM_SEED} \
                            --save_path ${save_path} \
                            --class_weight 1

wait 

# To train model C
python train_classifier.py --dataset ${DATASET} \
                            --model ${MODEL} \
                            --n_epochs ${epoch} \
                            --data_root ${DATA_ROOT} \
                            --manualSeed ${RANDOM_SEED} \
                            --save_path ${save_path} \
                            --class_weight 2
wait


