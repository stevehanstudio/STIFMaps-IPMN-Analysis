#/bin/bash

jupyter nbconvert --to notebook --execute preprocess_images.ipynb --ExcutePreprocessor.timeout=-1 --stdout

jupyter nbconvert --to notebook --execute gen_STIFMaps.ipynb --ExcutePreprocessor.timeout=-1 --stdout
