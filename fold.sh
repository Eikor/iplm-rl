fasta_path=$1
save_path=$2
for i in {0..3}; do
    sleep $i
    CUDA_VISIBLE_DEVICES=$i python dpo/divpro.py \
        ${fasta_path} \
        ${save_path}  &
done
wait
