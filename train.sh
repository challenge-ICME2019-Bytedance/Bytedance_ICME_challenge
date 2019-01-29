#!/bin/bash
cd  `dirname $0`
echo `pwd`
training_path=$1
validation_path=$2
echo "training path: " $training_path
echo "validation path: " $validation_path

save_model_dir=$3
echo "save model on: " $save_model_dir

batch_size=$4
embedding_size=$5
echo "batch size: " $batch_size
echo "embedding size: " $embedding_size

optimizer=$6
lr=$7

task=$8
track=$9
echo "task: " $task
echo "track: " $track

mkdir ${save_model_dir};

python train.py \
  --training_path $training_path \
  --validation_path $validation_path \
  --save_model_dir $save_model_dir \
  --batch_size $batch_size \
  --embedding_size $embedding_size \
  --lr $lr \
  --task $task \
  --track $track \
  --optimizer $optimizer
