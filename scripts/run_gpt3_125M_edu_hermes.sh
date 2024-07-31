# GPT-3 (125M) repro, but using FineWeb
# 125M parameter model on 300B tokens
# note context length: 1024 -> 2048 for GPT-3
# => 6 * 125e6 * 300e9 = ~= 2.25e20 capability model
# 572,204 steps of 524,288 tokens/step => 300B
# on 8X A100 80GB SXM ($14/hr) steps in ~150ms/iter
# => training time 572,204 * 150ms ~= 24 hours ~= $336

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt3_125M_edu_hermes_v5"
done_file="$out_dir/DONE_00019622"

while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi

    mpirun -np 2 ./train_gpt2cu \
                -i "dev/data/edu_fineweb10B_hermes/edu_fineweb_hermes_train_*.bin" \
                -j "dev/data/edu_fineweb10B_hermes/edu_fineweb_hermes_val_*.bin" \
                -o $out_dir \
                -v 250 -s 1000 -g 144 \
                -h 1 \
                -b 16 -t 2048 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0006 \
                -q 0.1 \
                -u 700 \
                -n 500 \
                -nk 5 \
                -nm 2000 \
                -ge 1 \
                -sl 5.0 \
                -sg 5.0 \
                -y 1 \
                -x 19622 \
                -e "gpt3:c768"

    sleep 1
done
