ROUND_NUM=$1
ALPHA=$2
STEP=$3
SUBSET=$4
PDB_PATH=$5
TEMP=1.1

for i in $(seq 0 $ROUND_NUM); do
    echo "-----------------Round ${i} of ${ROUND_NUM}--------------------"
    run_name="DPO_round_${i}_xlarge_LoRA_r16_a16_beta0.5_alpha${ALPHA}_step${STEP}_tmalign_subset${SUBSET}_t1.1_p1_1e-5"
    mkdir multi_round/${run_name}
    if [ $i -eq 0 ]; then
        peft_resume=""
    else
        peft_resume="multi_round/DPO_round_$((i-1))_xlarge_LoRA_r16_a16_beta0.5_alpha${ALPHA}_step${STEP}_tmalign_subset${SUBSET}_t1.1_p1_1e-5"
    fi
    
    # First step: Generate dataset
    # generate and fold new training data and test data   
    echo "Generating and folding new training data and test data for round ${i}"
    date
    if [ $i -eq 0 ]; then
        echo generate round 0 pass empty to adapter_path
        bash generate.sh ${SUBSET} 1.1 - multi_round/rollout_round${i}_temp${TEMP}_fixlen_${SUBSET}
    else
        bash generate.sh ${SUBSET} 1.1 ${peft_resume} multi_round/rollout_round${i}_temp${TEMP}_fixlen_${SUBSET}
    fi
    wait
    bash fold.sh multi_round/rollout_round${i}_temp${TEMP}_fixlen_${SUBSET} multi_round/rollout_round${i}_temp${TEMP}_fixlen_${SUBSET}_pdb
    wait
    # Second step: calculate tmalign
    echo "Calculating TMalign for round ${i}"
    # ref pdb path, pred pdb path, save path
    python utils/tmalign.py ${PDB_PATH} multi_round/rollout_round${i}_temp${TEMP}_fixlen_${SUBSET}_pdb multi_round/rollout_round${i}_temp${TEMP}_fixlen_${SUBSET}_pdb
    
    # Third step: construct new dataset
    echo "Constructing multi_round/rollout_round${i}_temp${TEMP}_fixlen_${SUBSET}_balanced.pt for round ${i}"
    # subset tmalign path, save path
    python utils/build_dataset.py ${SUBSET} multi_round/rollout_round${i}_temp${TEMP}_fixlen_${SUBSET}_pdb.pkl multi_round/hfdata_round${i}_ESMFOLD_TMalign_subset${SUBSET}_t1.1_p1.pt
    
    wait
    echo "Running round ${i} with run name: ${run_name}"
    bash dpo.sh ${ALPHA} ${STEP} ${run_name} "${peft_resume}" multi_round/hfdata_round${i}_ESMFOLD_TMalign_subset${SUBSET}_t1.1_p1.pt
done

bash generate.sh 10 0.1 ${peft_resume} multi_round/rollout_round${i}_temp0.1_fixlen_10
bash fold.sh multi_round/rollout_round${i}_temp0.1_fixlen_10 multi_round/rollout_round${i}_temp0.1_fixlen_10_pdb
