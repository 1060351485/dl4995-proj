we have 3 models for this project.

1. baseline mode, we build the pipeline at the beginning stage
2. shuffle(permute), one method we tried to organize training data, run one tif for 1 epoch then another tif
3. all-in-one, put all tifs' patches together

models folder: 3 models

aucs folder: auc for 3 models, auc-fin-all-in-one-val folder is we tried different learning rate and decay in SGD with Nesterov Momentum

heatmaps folder: tif078 heatmap with baseline model and all-in-one model

codes:
TIF.py			->	TIF class, operations on tif 
train-all-in-one.py 	->	train all-in-one model
train-all-in-one-val.py	->	try different learning rate and dacay
train-shuffle.py	-> 	train shuffle(permute) model
train-shuffle-more-epoch.py 	-> some tifs haven't converge yet, train more epoch


test-fin-all.py		->	test all-in-one model, print auc and heatmap
test-fin-base.py	-> 	test baseline model, print auc and heatmap
test-fin-shuffle.py	-> 	test shuffle(permute) model, print auc and heatmap