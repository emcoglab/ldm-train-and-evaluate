library(gdata)

spp_data_path = '/Users/caiwingfield/evaluation/tests/Semantic priming project/Hutchinson et al. (2013) SPP.xls'

spp_data = read.xls(spp_data_path)

n = names(spp_data)

print(n)

