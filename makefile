glove:
	wget -P ./data/ "http://nlp.stanford.edu/data/glove.6B.zip"
	unzip ./data/glove.6B.zip -d data/glove.6B/
#	rm ./data/glove.6B.zip

run:
	CUDA_VISIBLE_DEVICES=2 python build_data.py
	CUDA_VISIBLE_DEVICES=2 python train.py
	CUDA_VISIBLE_DEVICES=2 python evaluate.py
