total=$1
temperature=$2
adapter_path=$3
save_prefix=$4

structure_path=path_to_structure_embedding
model_path=your_local_model_path

for i in {0..3}; do
    sleep $i
    CUDA_VISIBLE_DEVICES=$i python utils/generate.py \
    --total ${total} \
    --num_return_sequences 2 \
    --temperature ${temperature} \
    --fix-length \
    --top_p 1.0 \
    --structure_path ${structure_path} \
    --model_path ${model_path} \
    --adapter_path ${adapter_path} \
    --save_prefix ${save_prefix} \
    --max_length 512 &
done

wait

python clean_generate.py ${save_prefix}

CUDA_VISIBLE_DEVICES=0 python utils/generate.py \
    --total ${total} \
    --num_return_sequences 2 \
    --temperature ${temperature} \
    --fix-length \
    --structure_path ${structure_path} \
    --model_path ${model_path} \
    --top_p 1.0 \
    --adapter_path ${adapter_path} \
    --save_prefix ${save_prefix} \
    --max_length 512 

