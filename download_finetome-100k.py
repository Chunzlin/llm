from modelscope.msdatasets import MsDataset

# 将 '/your/custom/path' 替换为你想要的任何本地路径
custom_cache_path = '../datasets/FineTome-100k/'
ds =  MsDataset.load('AI-ModelScope/FineTome-100k', cache_dir=custom_cache_path)
print(ds)