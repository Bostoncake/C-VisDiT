currenttime=`date "+%Y%m%d_%H%M%S"`
config=config/office/A-D-1.json                         # choose the corresponding configurations in `./config`
exp_id=${currenttime}"_Office-31_Amazon-DSLR-1-shot"    # rename the experiment folder to distinguish between experiments
CUDA_VISIBLE_DEVICES=0 python src/run.py \
    --config ${config} \
    --exp_id ${exp_id}
