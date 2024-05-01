import os 
dataset_dir=os.path.join(os.path.dirname(__file__),'data')
assert os.path.exists(dataset_dir), "Data directory not found"