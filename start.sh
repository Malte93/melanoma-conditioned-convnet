#!/bin/bash
: '
This shell script is intended to simplify the training process, as numerous 
experiments need to be conducted for the thesis. Each function defines a main 
experiment with several sub-experiments, where the only differences between 
the sub-experiments are the position and number of the FiLM layers.

Callable functions within this script:
    * vanilla_network
    * dropout_active
    * batchnorm_active
    * dropout_batchnorm_active
    * weighted_decay_active
    * pretrained

The first keyword argument determines the function you want to call,
e.g. start.ssh 'vanilla_network' (with quotation marks).

Feel free to add your own functions that define specific expermients.
  '

# Define global variables
PATH_DATASET = /Your/Path/Here/Dataset/
PATH_EXPERIMENTS = /Your/Path/Here/Experiments/

vanilla_network() {
    : '
        Run on different seeds:
        Run 1: start.sh 'vanilla_network' 'Vanilla_network_seed_42' 42 0.0003 'Adam'
        Run 2: start.sh 'vanilla_network' 'Vanilla_network_seed_43' 43 0.0003 'Adam'
        Run 3: start.sh 'vanilla_network' 'Vanilla_network_seed_44' 44 0.0003 'Adam'
    ' 
    # Variables
    EXPERIMENT_NAME=$2
    SEED=$3
    LEARNING_RATE=$4
    OPTIMIZER=$5
    BATCH_SIZE=32
    EPOCHS=20
    LOSS_FUNCTION='BCEWithLogitLoss'

    # Loop through experiments
    for i in $(seq 1 10); do
        echo "Training process starts for $EXPERIMENT_NAME! (Sub)Experiment number $i starts!"

        # Generate the film_layer argument dynamically
        if [ $i -le 9 ]; then
            FILM_LAYER=$(seq -s ' ' 1 $i)
        else
            FILM_LAYER="6 7 8 9"
        fi

        # Run the experiment
        python train.py --path_dataset $PATH_DATASET \
                        --path_experiment $PATH_EXPERIMENTS \
                        --experiment_name $EXPERIMENT_NAME \
                        --seed $SEED \
                        --experiment_number "number_00$i" \
                        --learning_rate $LEARNING_RATE \
                        --optimizer $OPTIMIZER \
                        --batch_size $BATCH_SIZE \
                        --epochs $EPOCHS \
                        --loss_function $LOSS_FUNCTION \
                        --weighted_random_sampler \
                        --film_layer $FILM_LAYER \

        echo "(Sub)Experiment number $i done!"
    done
}

dropout_active() {
    : '
        Run on different seeds:
        Run 1: start.sh 'dropout_active' 'dropout_seed_42' 42 0.0003 'Adam'
        Run 2: start.sh 'dropout_active' 'dropout_seed_43' 43 0.0003 'Adam'
        Run 3: start.sh 'dropout_active' 'dropout_seed_44' 44 0.0003 'Adam'
    '
    # Variables
    EXPERIMENT_NAME=$2
    SEED=$3
    LEARNING_RATE=$4
    OPTIMIZER=$5
    BATCH_SIZE=32
    EPOCHS=20
    LOSS_FUNCTION='BCEWithLogitLoss'

    # Loop through experiments
    for i in $(seq 1 10); do
        echo "Training process starts for $EXPERIMENT_NAME! (Sub)Experiment number $i starts!"

        # Generate the film_layer argument dynamically
        if [ $i -le 9 ]; then
            FILM_LAYER=$(seq -s ' ' 1 $i)
        else
            FILM_LAYER="6 7 8 9"
        fi

        # Run the experiment
        python train.py --path_dataset $PATH_DATASET \
                        --path_experiment $PATH_EXPERIMENTS \
                        --experiment_name $EXPERIMENT_NAME \
                        --seed $SEED \
                        --experiment_number "number_00$i" \
                        --learning_rate $LEARNING_RATE \
                        --optimizer $OPTIMIZER \
                        --batch_size $BATCH_SIZE \
                        --epochs $EPOCHS \
                        --loss_function $LOSS_FUNCTION \
                        --weighted_random_sampler \
                        --film_layer $FILM_LAYER \
                        --dropout_active \

        echo "(Sub)Experiment number $i done!"
    done
}

batchnorm_active() {
    : '
        Run on different seeds:
        Run 1: start.sh 'batchnorm_active' 'batchnorm_seed_42' 42 0.0003 'Adam'
        Run 2: start.sh 'batchnorm_active' 'batchnorm_seed_43' 43 0.0003 'Adam'
        Run 3: start.sh 'batchnorm_active' 'batchnorm_seed_44' 44 0.0003 'Adam'
    '
    # Variables
    EXPERIMENT_NAME=$2
    SEED=$3
    LEARNING_RATE=$4
    OPTIMIZER=$5
    BATCH_SIZE=32
    EPOCHS=20
    LOSS_FUNCTION='BCEWithLogitLoss'

    # Loop through experiments
    for i in $(seq 1 10); do
        echo "Training process starts for $EXPERIMENT_NAME! (Sub)Experiment number $i starts!"

        # Generate the film_layer argument dynamically
        if [ $i -le 9 ]; then
            FILM_LAYER=$(seq -s ' ' 1 $i)
        else
            FILM_LAYER="6 7 8 9"
        fi

        # Run the experiment
        python train.py --path_dataset $PATH_DATASET \
                        --path_experiment $PATH_EXPERIMENTS \
                        --experiment_name $EXPERIMENT_NAME \
                        --seed $SEED \
                        --experiment_number "number_00$i" \
                        --learning_rate $LEARNING_RATE \
                        --optimizer $OPTIMIZER \
                        --batch_size $BATCH_SIZE \
                        --epochs $EPOCHS \
                        --loss_function $LOSS_FUNCTION \
                        --weighted_random_sampler \
                        --film_layer $FILM_LAYER \
                        --batchnorm_active \

        echo "(Sub)Experiment number $i done!"
    done
}

dropout_batchnorm_active() {
    : '
        Run on different seeds:
        Run 1: start.sh 'dropout_batchnorm_active' 'dropout_batchnorm_seed_42' 42 0.0003 'Adam'
        Run 2: start.sh 'dropout_batchnorm_active' 'dropout_batchnorm_seed_43' 43 0.0003 'Adam'
        Run 3: start.sh 'dropout_batchnorm_active' 'dropout_batchnorm_seed_44' 44 0.0003 'Adam'
    '
    # Variables
    EXPERIMENT_NAME=$2
    SEED=$3
    LEARNING_RATE=$4
    OPTIMIZER=$5
    BATCH_SIZE=32
    EPOCHS=20
    LOSS_FUNCTION='BCEWithLogitLoss'

    # Loop through experiments
    for i in $(seq 1 10); do
        echo "Training process starts for $EXPERIMENT_NAME! (Sub)Experiment number $i starts!"

        # Generate the film_layer argument dynamically
        if [ $i -le 9 ]; then
            FILM_LAYER=$(seq -s ' ' 1 $i)
        else
            FILM_LAYER="6 7 8 9"
        fi

        # Run the experiment
        python train.py --path_dataset $PATH_DATASET \
                        --path_experiment $PATH_EXPERIMENTS \
                        --experiment_name $EXPERIMENT_NAME \
                        --seed $SEED \
                        --experiment_number "number_00$i" \
                        --learning_rate $LEARNING_RATE \
                        --optimizer $OPTIMIZER \
                        --batch_size $BATCH_SIZE \
                        --epochs $EPOCHS \
                        --loss_function $LOSS_FUNCTION \
                        --weighted_random_sampler \
                        --film_layer $FILM_LAYER \
                        --dropout_active \
                        --batchnorm_active \

        echo "(Sub)Experiment number $i done!"
    done
}

weighted_decay_active() {
    : '
        Run on different seeds:
        Run 1: start.sh 'weighted_decay_active' 'weighted_decay_seed_42' 42 0.0003 'Adam' 0.00001
        Run 2: start.sh 'weighted_decay_active' 'weighted_decay_seed_43' 43 0.0003 'Adam' 0.00001
        Run 3: start.sh 'weighted_decay_active' 'weighted_decay_seed_44' 44 0.0003 'Adam' 0.00001
    '
    # Variables
    EXPERIMENT_NAME=$2
    SEED=$3
    LEARNING_RATE=$4
    OPTIMIZER=$5
    WEIGHTED_DECAY=$6
    BATCH_SIZE=32
    EPOCHS=20
    LOSS_FUNCTION='BCEWithLogitLoss'

    # Loop through experiments
    for i in $(seq 1 10); do
        echo "Training process starts! Experiment number $i starts!"

        # Generate the film_layer argument dynamically
        if [ $i -le 9 ]; then
            FILM_LAYER=$(seq -s ' ' 1 $i)
        else
            FILM_LAYER="6 7 8 9"
        fi

        # Run the experiment
        python train.py --path_dataset $PATH_DATASET \
                        --path_experiment $PATH_EXPERIMENTS \
                        --experiment_name $EXPERIMENT_NAME \
                        --seed $SEED \
                        --experiment_number "number_00$i" \
                        --learning_rate $LEARNING_RATE \
                        --optimizer $OPTIMIZER \
                        --weighted_decay $WEIGHTED_DECAY \
                        --batch_size $BATCH_SIZE \
                        --epochs $EPOCHS \
                        --loss_function $LOSS_FUNCTION \
                        --weighted_random_sampler \
                        --film_layer $FILM_LAYER \

        echo "Experiment number $i done!"
    done
}

pretrained() {
    : '
        Run on different seeds:
        Run 1: start.sh 'pretrained' 'pretrained_seed_42' 42 0.0003 'Adam'
        Run 2: start.sh 'pretrained' 'pretrained_seed_43' 43 0.0003 'Adam'
        Run 3: start.sh 'pretrained' 'pretrained_seed_44' 44 0.0003 'Adam'
    '
    # Variables
    EXPERIMENT_NAME=$2
    SEED=$3
    LEARNING_RATE=$4
    OPTIMIZER=$5
    BATCH_SIZE=32
    EPOCHS=20
    LOSS_FUNCTION='BCEWithLogitLoss'

    # Loop through experiments
    for i in $(seq 1 10); do
        echo "Training process starts! Experiment number $i starts!"

        # Generate the film_layer argument dynamically
        if [ $i -le 9 ]; then
            FILM_LAYER=$(seq -s ' ' 1 $i)
        else
            FILM_LAYER="6 7 8 9"
        fi

        # Run the experiment
        python train.py --path_dataset $PATH_DATASET \
                        --path_experiment $PATH_EXPERIMENTS \
                        --experiment_name $EXPERIMENT_NAME \
                        --seed $SEED \
                        --experiment_number "number_00$i" \
                        --learning_rate $LEARNING_RATE \
                        --optimizer $OPTIMIZER \
                        --batch_size $BATCH_SIZE \
                        --epochs $EPOCHS \
                        --loss_function $LOSS_FUNCTION \
                        --weighted_random_sampler \
                        --film_layer $FILM_LAYER \
                        --pretrained \

        echo "Experiment number $i done!"
    done
}

# Function call
if [ "$1" == "vanilla_network" ]; then
    vanilla_network $2 $3 $4 $5
elif [ "$1" == "dropout_batchnorm_active" ]; then
    dropout_batchnorm_active $2 $3 $4 $5
elif [ "$1" == "dropout_active" ]; then
    dropout_active $2 $3 $4 $5
elif [ "$1" == "batchnorm_active" ]; then
    batchnorm_active $2 $3 $4 $5
elif [ "$1" == "weighted_decay_active" ]; then
    weighted_decay_active $2 $3 $4 $5 $6
elif [ "$1" == "pretrained" ]; then
    pretrained $2 $3 $4 $5
else
    echo "Invalid function name"
fi 