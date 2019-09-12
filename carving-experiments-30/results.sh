cat results/*/experiments.tsv | grep _cat | sed -e 's/C64_16_2pr_C32_4_2pr_C64_32_2pr_F_D//' -e 's/_cat//' | awk '{print $1, $6}'

