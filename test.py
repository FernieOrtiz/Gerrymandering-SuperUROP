"""
Created on Thu Sep  6 15:14:38 2018
@author: daryl
"""
import matplotlib
matplotlib.use('TkAgg')

from new_seeds import *
#import tree_partition as tp
from districting_map import *
from districting_metrics import *
import json
import random
from make_graph import construct_graph
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gp
import pickle
import pprint

'''
print(df.columns)
Pennsylvania = construct_graph(df, pop_col="TOT_POP", district_col="VTDI10",
                        area_col = "ALAND10" ,data_cols = ["T16PRESD","T16PRESR","WHITE_POP","HISP_POP","BLACK_POP","ASIAN_POP"], data_source_type="geodataframe")
with open('/Users/fernandoortiz/Downloads/PA_VTD/PA_Graph.txt', 'wb') as f:
	pickle.dump(Pennsylvania,f)


df = gp.read_file("MA_precincts_12_16/MA_precincts_12_16.shp")
print(df.columns)
df.plot(column= "OBJECTID", cmap = "tab20")
plt.show()
'''
'''
df = gp.read_file("PA_VTD/PA_VTD.shp")
df.plot(column= "2011_PLA_1", cmap = "tab20")
plt.show()
c
print(df.columns)
Pennsylvania = construct_graph(df, pop_col="TOT_POP", district_col="VTDI10",
                        area_col = "ALAND10" ,data_cols = ["T16PRESD","T16PRESR","WHITE_POP","HISP_POP","BLACK_POP","ASIAN_POP"], data_source_type="geodataframe")
with open('/Users/fernandoortiz/Downloads/PA_VTD/PA_Graph.txt', 'wb') as f:
	pickle.dump(Pennsylvania,f)
'''

'''
df = gp.read_file("IA_counties/IA_counties.shp")
print(df.columns)
cols = list(df.columns)
cols.remove("geometry")
pop_col = "TOTPOP"
district_col = "CD"
cols.remove(pop_col)
cols.remove(district_col)
df.plot(column= "CD", cmap = "tab20")
plt.show()
Iowa = construct_graph(df, pop_col= pop_col, district_col= district_col,
                        data_cols = cols, data_source_type="geodataframe")
with open('/Users/fernandoortiz/Downloads/Gerrymandering_Project/IA_counties/IA_counties.txt', 'wb') as f:
	pickle.dump(Iowa,f)
print("Done with IA")
'''
df = gp.read_file("IA_counties/IA_counties.shp")
df["ID"] = df.index
IA_Graph_pickle = open('IA_counties/IA_counties.txt',"rb")
IA = pickle.load(IA_Graph_pickle)
IA_Map = Districting_Map(IA, "ID", df)
IA_Map.add_attribute_percentages(["PRES16D","PRES16R"])
IA_Map.set_partition_to_column("CD")
#IA_Map.plot_districting(legend=False)
#IA_Map.plot_districts_by_demographic("PRES16D_PCT", colors = "Blues")
print(IA_Map.evaluate_on_metric(outgoing_edge_score))
IA_Map.outlier_analysis_global_iteration(outgoing_edge_score, 100,.01,True)
IA_Map.optimize_global([outgoing_edge_score], [1], .01, 4, 10000, use_current_partition = True, view_scores = True)
IA_Map.plot_districting(legend=False)


df = gp.read_file("VA_precincts/VA_precincts.shp")
df["ID"] = df.index
VA_Graph_pickle = open('VA_precincts/VA_Graph.txt',"rb")
VA = pickle.load(VA_Graph_pickle)
VA_Map = Districting_Map(VA, "ID", df)
VA_Map.set_partition_to_column("CD_12")
VA_Map.plot_districting(legend = False)
print(VA_Map.efficiency_gap("G16DPRS","G16RPRS"))
VA_Map.set_partition_to_column("CD_16")
VA_Map.plot_districting(legend = False)
print(VA_Map.efficiency_gap("G16DPRS","G16RPRS"))
VA_Map.plot_districts_by_demographic("G16RPRS_PCT")
cost = Linear_Districting_Cost([outgoing_edge_score], [1], population_eps = .01)
cost.pairwise_optimizer(VA_Map, 100, 30, number_districts = 2, use_current_partition = True, view_scores = True)
VA_Map.plot_districting(legend = False)

'''
df = gp.read_file("Gerrymandering_Project/NC_VTD/NC_VTD.shp")
df["ID"] = df.index
NC_Graph_pickle = open('/Users/fernandoortiz/Downloads/Gerrymandering_Project/NC_VTD/NC_Graph.txt',"rb")
NC = pickle.load(NC_Graph_pickle)
dto.numbers_into_percents(NC, ["EL16G_PR_D","EL16G_PR_R"])
NC_Graph = dto.Districting_Graph(NC)
print(NC_Graph.graph[0])
NC_Map = dto.Districting_Map(NC_Graph, "ID", df)
NC_Map.set_partition_to_column("oldplan")
NC_Map.plot_districting(legend = False)
#NC_Map.plot_districts_by_demographic("EL16G_PR_D_PCT",colors = "Blues")
print(NC_Map.efficiency_gap("EL16G_PR_D","EL16G_PR_R"))
NC_Map.set_partition_to_column("newplan")
NC_Map.plot_districting(legend = False)
#NC_Map.plot_districts_by_demographic("EL16G_PR_D_PCT",colors = "Blues")
print(NC_Map.efficiency_gap("EL16G_PR_D","EL16G_PR_R"))
cost = dto.Linear_Districting_Cost([dto.outgoing_edge_score], [1], population_eps = .01)
cost.pairwise_optimizer(NC_Map, 5, 2, number_districts = 2, use_current_partition = True, view_scores = True)
egs = []
for i in range(10):
	NC_Map.set_partition_to_column("newplan")
	cost.pairwise_optimizer(NC_Map, 25, 20, number_districts = 2, use_current_partition = True, view_scores = True)
	egs.append(NC_Map.efficiency_gap("EL16G_PR_D","EL16G_PR_R"))

print(NC_Map.efficiency_gap("EL16G_PR_D","EL16G_PR_R"))
#NC_Map.plot_districting()
NC_Map.plot_districts_by_demographic("EL16G_PR_R")
'''

'''
df = gp.read_file("PA_VTD/NC_VTD/NC_VTD.shp")
df.plot(column= "2011_PLA_1", cmap = "tab20")
plt.show()

print(df.columns)
Pennsylvania = construct_graph(df, pop_col="TOT_POP", district_col="VTDI10",
                        area_col = "ALAND10" ,data_cols = ["T16PRESD","T16PRESR","WHITE_POP","HISP_POP","BLACK_POP","ASIAN_POP"], data_source_type="geodataframe")
with open('/Users/fernandoortiz/Downloads/PA_VTD/PA_Graph.txt', 'wb') as f:
	pickle.dump(Pennsylvania,f)
'''

df = gp.read_file("PA_VTD/PA_VTD.shp")
df["ID"] = df.index
#df["BLACK"] = df["BLACK_POP"]/(df["WHITE_POP"]+df["BLACK_POP"]+df["ASIAN_POP"]+df["HISP_POP"])
#df.plot(column = "BLACK", cmap = "Greens", legend = True)
PA_Graph_pickle = open('/Users/fernandoortiz/Downloads/PA_VTD/PA_Graph.txt',"rb")
PA = pickle.load(PA_Graph_pickle)
dto.numbers_into_percents(PA, ["T16PRESD","T16PRESR"])
PA_Graph = dto.Districting_Graph(PA)
PA_Map = dto.Districting_Map(PA_Graph, "ID", df)
PA_Map.set_partition_to_column("2011_PLA_1")
PA_Map.plot_districting(legend = False)
print(PA_Map.efficiency_gap("T16PRESD","T16PRESR"))
PA_Map.set_partition_to_column("REMEDIAL_P")
PA_Map.plot_districting(legend = False)
print(PA_Map.efficiency_gap("T16PRESD","T16PRESR"))
init_breakdown = PA_Map.district_demographic_breakdown("T16PRESR_PCT")
PA_Map.plot_districts_by_demographic("T16PRESR_PCT")
PA_population = sum([PA_Graph.graph.nodes[node]["population"] for node in PA_Graph.graph.nodes])
#cost = tp.Linear_Districting_Cost([lambda graph, partitions: tp.population_balance_score(graph, partitions, 2, target = PA_population/k), tp.outgoing_edge_score], 
#									[10,100], population_eps = .05)
cost = dto.Linear_Districting_Cost([dto.outgoing_edge_score], [1], population_eps = .01)
#cost = dto.Linear_Districting_Cost([lambda graph, partitions: dto.demographic_majority_districts_score(graph, partitions, "T16PRESD_PCT"),
#									dto.outgoing_edge_score],
#									lambda graph, partitions: dto.district_packing_score(graph, partitions, "T16PRESD_PCT")],
#									[1,1000],population_eps = .01)
# PA_Map.plot_districting()
cost.pairwise_optimizer(PA_Map, 30, 25, number_districts = 2, use_current_partition = True, view_scores = True)
pprint.pprint(init_breakdown)
pprint.pprint(PA_Map.district_demographic_breakdown("T16PRESR_PCT"))
PA_Map.plot_districting(legend = False)
PA_Map.plot_districts_by_demographic("T16PRESR_PCT")
print(PA_Map.efficiency_gap("T16PRESD","T16PRESR"))



'''
graph_path ="Arkansas_graph_with_data.json"
Arkansas = construct_graph(graph_path, id_col="ID",  pop_col="POP10", district_col="CD",
                        data_source_type="json")
df = gp.read_file("AR_Full/AR_Full.shp")
k = 3
AK_graph = tp.Districting_Graph(Arkansas)
AK_Map = tp.Districting_Map(AK_graph, "ID", df)
AK_population = sum([AK_graph.graph.nodes[node]["population"] for node in AK_graph.graph.nodes])
cost = tp.Linear_Districting_Cost([lambda graph, partitions: tp.population_balance_score(graph, partitions, k, target = AK_population/k), tp.outgoing_edge_score], 
									[10,100], population_eps = .10)
AK_Map.set_rand_partition(k)
AK_Map.plot_districting()
cost.tree_sample_optimizer(AK_Map, k, 1, use_current_partition = True)
AK_Map.plot_districting()
cost.pairwise_optimizer(AK_Map, 20, 10, use_current_partition = True)
AK_Map.plot_districting()
'''

'''
graph_path ="Arkansas_graph_with_data.json"
df = gp.read_file("AR_Full/AR_Full.shp")
#df["partition"] = 0
#df=df.set_index("ID")
print(df.shape)
print(df.columns)

Arkansas = construct_graph(graph_path, id_col="ID",  pop_col="POP10", district_col="CD",
                        data_source_type="json")
print("HERE")
k = 3
AK_graph = tp.Districting_Graph(Arkansas)
print(AK_graph.zero_population_blocks())
AK_Map = tp.Districting_Map(AK_graph, "ID", df)
AK_population = sum([AK_graph.graph.nodes[node]["population"] for node in AK_graph.graph.nodes])
cost = tp.Linear_Districting_Cost([lambda graph, partitions: tp.population_balance_score(graph, partitions, k, target = AK_population/k), tp.outgoing_edge_score], 
									[10,100], population_eps = .10)
AK_Map.set_rand_partition(k)
AK_Map.plot_districting()

#AK_Map.set_partition_graph()

cost.tree_sample_optimizer(AK_Map, 3, 10, use_current_partition = True)
AK_Map.plot_districting()
'''

'''
AK_Map.plot_districting()
cost.pairwise_optimizer(AK_Map, 10, 10, use_current_partition = True)
'''



graph_path ="Arkansas_graph_with_data.json"
Arkansas = construct_graph(graph_path, id_col="ID",  pop_col="POP10", district_col="CD",
                        data_source_type="json")
df = gp.read_file("AR_Full/AR_Full.shp")
AK = dto.Districting_Graph(Arkansas)
AK_Map = dto.Districting_Map(AK, "ID", df)
AK.number_partitions_histogram(1000, "Arkansas", 2, .01)
k = 4
cost = dto.Linear_Districting_Cost([dto.outgoing_edge_score], [1], population_eps = .01)
AK_Map.set_rand_partition(k, .01)
AK_Map.plot_districting()
cost.tree_sample_optimizer(AK_Map, k, 25, use_current_partition = True)
AK_Map.plot_districting()
cost.pairwise_optimizer(AK_Map, 50, 10, use_current_partition = True)
AK_Map.plot_districting(legend = False)

#AK.lowest_admissible_epsilon_histogram(100, "Arkansas", 2)
#AK.number_partitions_histogram(1000, "Arkansas", 2, .01)



