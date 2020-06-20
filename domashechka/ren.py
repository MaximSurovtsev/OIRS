import os 
import glob

fnames = list(glob.glob('GENERATED/*.png'))
print(fnames)
for i, f in enumerate(fnames):
	os.rename(f, f'GENERATED/image{i}.png')