# GPT-3 (124M) repro on FineWeb
# 124M parameter model on 300B tokens
# note context length: 1024 -> 2048 for GPT-3
# => 6 * 124e6 * 300e9 = 7.44e18 ~= 2.2e20 capability model
# 565,950 steps of 524,288 tokens/step
# on 8X A100 80GB SXM ($14/hr) steps in ~300ms/iter
# => training time 565,950 * 300ms ~= 47 hours ~= $658

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt2_350M_edu_hermes"
done_file="$out_dir/DONE_00019622"

while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi

    # run python dev/data/fineweb_edu_hermes.py --version 10B to prepro data
    # run python dev/data/hellaswag.py to prepro hellaswag eval
    mpirun -np 2 ./train_gpt2cu \
                -i "dev/data/edu_fineweb10B_hermes/edu_fineweb_hermes_train_*.bin" \
                -j "dev/data/edu_fineweb10B_hermes/edu_fineweb_hermes_val_*.bin" \
                -o $out_dir \
                -v 250 -s 5000 -g 144 \
                -h 1 \
                -b 16 -t 1024 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0003 \
                -q 0.0 \
                -u 700 \
                -n 5000 \
                -sl 7.0 -sg 7.0 \
                -y 1 \
                -x 19622 \
                -e "d24"

    sleep 1
done
