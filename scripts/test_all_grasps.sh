cate=$1
pkl_root=$2
suff=$3
results_name=$4
mode='eval'

for inst_name in `ls ${pkl_root}/${mode}`
do
    echo ${inst_name}
    for part in `ls ${pkl_root}/${mode}/${inst_name}`
    do
        echo ${inst_name}/${part}
        for v in `ls ${pkl_root}/${mode}/${inst_name}/${part}`
        do
            result=$(echo $v | grep $suff)
            if [ "$result" != '' ]; then
                python isaac_sim/sim_one_object_all_grasps.py \
                --category ${cate} \
                --pkl_file ${inst_name}/${part}/${v} \
                --pkl_root ${pkl_root} \
                --mode ${mode} \
                --results_filename $results_name
            fi
        done
    done
done