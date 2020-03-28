python main.py --name exp_dcgan_bs32 --type dcgan --spectral_norm False --epochs 750 --batch_size 32
python main.py --name exp_dcgan_spectral_bs32 --type dcgan --spectral_norm True --epochs 750 --batch_size 32

python main.py --name exp_acgan_type_1_bs32 --type acgan --spectral_norm False --epochs 750 --batch_size 32 --label_column type_1
python main.py --name exp_acgan_shape_bs32 --type acgan --spectral_norm False --epochs 750 --batch_size 32 --label_column shape

python main.py --name exp_acgan_type_1_spectral_bs32 --type acgan --spectral_norm True --epochs 750 --batch_size 32  --label_column type_1
python main.py --name exp_acgan_shape_spectral_bs32 --type acgan --spectral_norm True --epochs 750 --batch_size 32 --label_column shape
