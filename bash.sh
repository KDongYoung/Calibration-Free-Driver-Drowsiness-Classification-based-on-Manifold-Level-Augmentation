echo "Public dataset"
for idx in 0 1 2 3 4 5 6 7 # GPU IDX
do
    gpu_idx=$(($idx%4))                                                                                                                                                                                                                                                                              
    CUDA_VISIBLE_DEVICES=$gpu_idx python code정리/TotalMain.py --subject_group=$idx --cuda_num=$gpu_idx --dataset_name=SA&
done
wait
echo "Public dataset End"