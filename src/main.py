import argparse

parser = argparse.ArgumentParser(description='Main script')

parser.add_argument('--data_dir', type=str, default='C:/Users/Jonas/Documents/GitHub/pokemon-generation/data/sprites')

parser.add_argument('--name', type=str, default='gan')
parser.add_argument('--type', type=str, default='dcgan', help='GAN Type')

parser.add_argument('--spectral_norm', type=str, default=False)

parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--sample_interval', type=int, default=5)

parser.add_argument('--label_column', type=str, default='type_1')

args = parser.parse_args()

from dcgan import DCGAN
from acgan import ACGAN

config = {'spectral_norm': args.spectral_norm}

if args.type == 'dcgan':
    model = DCGAN(name=args.name, config=config)
elif args.type == 'acgan':
    model = ACGAN(name=args.name, label_column=args.label_column, config=config)

model.fit(args.data_dir, args.epochs, args.batch_size, args.sample_interval)
