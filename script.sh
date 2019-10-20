export PYTHONPATH=$PYTHONPATH:$(pwd)

python main.py --shared layer3 --group_norm 32 --workers 16 --outf results/resnet18_layer3_gn

CUDA_VISIBLE_DEVICES=0 python script_test.py 5 layer3 online_shuffle gn
CUDA_VISIBLE_DEVICES=0 python script_test.py 5 layer3 slow gn
CUDA_VISIBLE_DEVICES=0 python test_calls/test_video.py --shared layer3 --group_norm 32 --niter 1 \
				--resume results/resnet18_layer3_gn --outf results/triter1_layer3_gn_video

python main.py --shared none --group_norm 32 --workers 16 --outf results/resnet18_layer3_gn
CUDA_VISIBLE_DEVICES=0 python script_test.py 5 none none gn
