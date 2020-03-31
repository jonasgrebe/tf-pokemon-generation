python main.py --name exp_dcgan_sn_bs32 --type dcgan --epochs 250 --batch_size 32 --spectral_norm
python main.py --name exp_dcgan_bs32 --type dcgan --epochs 250 --batch_size 32

python main.py --name exp_acgan_sn_bs32 --type acgan --epochs 250 --batch_size 32 --spectral_norm --label_column type_1
python main.py --name exp_acgan_bs32 --type acgan --epochs 250 --batch_size 32 --label_column type_1
