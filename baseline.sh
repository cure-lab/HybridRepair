export CUBLAS_WORKSPACE_CONFIG=:16:8

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

DATE=`date +%Y-%m-%d`
echo $DATE
DIRECTORY=./save/${DATE}/
if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configuration   ##############################

DATA_ROOT='YOUR DATASET'

#  ------Test Center:  Selection ------
DATASET='cifar10'
# DATASET='svhn'
# DATASET='gtsrb'

# default parameters
weight_decay=5e-4
epochs=10
lr=0.001
STEP=100
####  ------Test Center:  Selection Version 2, fixed mini-budget ------
# run testing

for avg_clustersize in 10; do 
for p_budget in 1; do # 1, 5, 10
for RANDOM_SEED in 10; do # 10, 20, 30
for exp in 0; do # 0 represents cut the unlabeled data > tau; -1 represents all unlabeled data are selected.
for lam in 100; do # The ratio between labeled and unlabeled data: default=100
for MODEL in 'MobileNet'; do # MobileNet, resnet18, ShuffleNetG2
    for MODEL_NO in 0; do # 0, 1, 2 represents Model A, B, C
        for SOLUTION in 'mcp'; do # 'gini' 'coreset' 'badge' 'SSLConsistency' 'SSLConsistency-Imp' 'SSLRandom'
            for eps in 0.3;do
            for min_samples in 3; do # default=3
                for tao in 0.1; do
                    clusteralg='hybrid'
                    fe='model2test'

                    echo 'tao='$tao
                    echo $SOLUTION
                    echo $MODEL_NO
                    echo $eps
                    echo $clusteralg
                    echo $fe
                    echo $avg_clustersize

                    MODEL2TEST=${MODEL}
                    MODEL2TESTPATH=./checkpoint/${DATASET}/ckpt_bias/${MODEL}_${MODEL_NO}_b.t7 
                    save_path=save/${DATE}/${DATASET}_${MODEL2TEST}_${STEP}
                    echo 'model to test arch '$MODEL2TEST
                    echo 'model to test path '$MODEL2TESTPATH
                    python selection.py \
                            --dataset $DATASET \
                            --manualSeed ${RANDOM_SEED} \
                            --model2test_arch $MODEL2TEST \
                            --model2test_path $MODEL2TESTPATH \
                            --model_number $MODEL_NO \
                            --step ${STEP} \
                            --p_budget ${p_budget} \
                            --save_path ${save_path} \
                            --data_path ${DATA_ROOT} \
                            --solution ${SOLUTION} \
                            --retrain_lr ${lr} \
                            --retrain_weightdecay ${weight_decay} \
                            --retrain_epoch ${epochs} \
                            --retrain \
                            --exp $exp \
                            --eps $eps \
                            --lam_ul $lam \
                            --min_samples $min_samples \
                            --fe $fe \
                            --cluster $clusteralg \
                            --avg_clustersize ${avg_clustersize} \
                            --tao $tao \
                            --u_weight \
                            --sel_l_data 
                done
                done
                done
                done
            done
            done
            done
            done
        done
    done
done



