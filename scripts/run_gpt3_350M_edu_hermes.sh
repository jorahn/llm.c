# GPT-3 (350M) training on FineWeb-Edu + OpenHermes 2.5
# 350M parameter model on 10B tokens
# note context length: 1024 -> 2048 for GPT-3
# => 6 * 350e6 * 10e9 ~= 2.1e19 capability model
# 39,125 steps of 270,926 tokens/step
# on 2X RTX4090 24GB steps in ~2,800ms/iter
# => training time 39,125 * 2,800ms ~= 30 hours

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt3_350M_edu_hermes"
done_file="$out_dir/DONE_00039125"
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
                -d 262144 \
                -r 0 \
                -z 0 \
                -c 0.1 \
                -l 0.0006 \
                -q 0.1 \
                -u 700 \
                -n 1000 -nk 5 -nm 5000 \
                -sl 7.0 -sg 7.0 \
                -y 1 \
                -x 39125 \
                -e "gpt3:c1024"

    sleep 1
done
