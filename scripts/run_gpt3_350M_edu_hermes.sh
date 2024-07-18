# GPT-3 (350M) repro on FineWeb-Edu
# 350M parameter model on 10B tokens
# note context length: 1024 -> 2048 for GPT-3
# => 6 * 350e6 * 10e9 ~= 2.1e19 capability model
# 313,004 steps of 31,948 tokens/step
# on 2X RTX4090 24GB steps in ~500ms/iter
# => training time 313,004 * 500ms ~= 44 hours

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt3_350M_edu_hermes"
done_file="$out_dir/DONE_00313004"
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
                -v 2000 -s 10000 -g 144 \
                -h 1 \
                -b 8 -t 2048 \
                -d 32768 \
                -r 0 \
                -z 0 \
                -c 0.1 \
                -l 0.0006 \
                -q 0.0 \
                -u 500 \
                -n 10000 \
                -y 1 \
                -e "gpt3:c1024"

    sleep 1
done
