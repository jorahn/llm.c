# GPT-3 (356M) training on FineWeb-Edu + OpenHermes 2.5
# 356M parameter model on 11B tokens
# note context length: 1024 -> 2048 for GPT-3
# => 6 * 356e6 * 11e9 ~= 2.3e19 capability model
# 19,577 steps of 561,884 tokens/step
# on 2X RTX4090 24GB steps in ~5,500ms/iter
# => training time 19,577 * 5,500ms ~= 30 hours

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt3_350M_edu_hermes"
done_file="$out_dir/DONE_00019577"
while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi

    # run python dev/data/edu_fineweb_hermes.py to prepro data
    # run python dev/data/hellaswag.py to prepro hellaswag eval
    mpirun -np 2 ./train_gpt2cu \
                -i "dev/data/edu_fineweb10B_hermes/edu_fineweb_hermes_train_*.bin" \
                -j "dev/data/edu_fineweb10B_hermes/edu_fineweb_hermes_val_*.bin" \
                -o $out_dir \
                -v 250 -s 1000 -g 144 \
                -h 1 \
                -b 8 -t 2048 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0003 \
                -q 0.0 \
                -u 700 \
                -n 1000 \
                -sl 7.0 -sg 7.0 \
                -y 1 \
                -x 19577 \
                -e "gpt3:c1024"

    sleep 1
done
