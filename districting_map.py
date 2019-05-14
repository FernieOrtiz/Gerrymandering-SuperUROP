import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np
import make_graph 
import geopandas as gp
import itertools
from networkx.algorithms import tree
import pprint
import tree_processing as trees
from districting_optimizer import *
from outlier_analysis import *


def get_spanning_tree(graph, node_communities):
	'''
	Returns a pseudo-random spanning tree of the input graph
	'''
	w=graph.copy()
	h=graph.copy()
	community_edges = set()
	for community in node_communities:
		community_subgraph = graph.subgraph(community)
		community_edges.update(community_subgraph.edges())
	for ed in w.edges():
		if ed not in community_edges:
			w.add_edge(ed[0],ed[1],weight=random.random())
		else:
			print("In Community")
			w.add_edge(ed[0],ed[1],weight=random.random()+100)
	T = [(edge[0],edge[1]) for edge in tree.maximum_spanning_edges(w, algorithm='kruskal', data=True)]
	for edge in graph.edges():
		if edge not in T and (edge[1],edge[0]) not in T:
			h.remove_edge(edge[0],edge[1])
	can_cut_dict = {edge:(False if edge in community_edges else True) for edge in h.edges()}
	nx.set_edge_attributes(h, can_cut_dict, 'cut')
	return h


class Districting_Graph:
	'''
	Object that takes in a Network


	''' 


	def __init__(self,graph):
		self.graph = graph
		self.tree = None
		self.node_communities = []

	def zero_population_blocks(self):
		return len([node for node in self.graph.nodes if self.graph.nodes[node]["population"] == 0])

	def reset_node_communities(self):
		self.node_communities = []

	def add_node_community(self,node_community):
		self.node_communities.append(node_community)

	def set_node_community(self,node_communities):
		self.node_communities = node_communities

	def generate_rand_tree(self):
		self.tree = get_spanning_tree(self.graph, self.node_communities)

	def partition_by_tree(self,cut_edges):
		return trees.partition_subgraphs(self.tree,cut_edges)

	def lowest_tree_epsilon(self, k, delta = .001):
		curr_epsilon = 1
		if not trees.balanced_population_partition_exists(self.tree,k,curr_epsilon):
			return "No Balanced Partitions Possible"
		prev_epsilon = 1
		curr_epsilon = .5
		smallest_epsilon = 1
		count = 1
		while abs(curr_epsilon - prev_epsilon) > delta:
			#print(curr_epsilon)
			count += 1
			if trees.balanced_population_partition_exists(self.tree,k,curr_epsilon):
				smallest_epsilon = curr_epsilon
				prev_epsilon = curr_epsilon
				curr_epsilon -= .5**count
			else:
				prev_epsilon = curr_epsilon
				curr_epsilon += .5**count
		return smallest_epsilon

	def lowest_admissible_epsilon_histogram(self, samples, name, k, initial_eps = 1, bins = 20, interval = (0, 0.4)):
		eps = []
		for i in range(samples):
			self.generate_rand_tree()
			eps.append(self.lowest_tree_epsilon(k))
		plt.hist(eps,bins = bins,range = interval, density = True)
		plt.title("Lowest Admissible Epsilon Histogram: " + name)
		plt.show()

	def number_partitions_histogram(self, samples, name, k, eps, bins = 20):
		num_partitions = []
		for i in range(samples):
			self.generate_rand_tree()
			num_partitions.append(len(trees.balanced_population_partition(self.tree,k,eps)))
		plt.hist(num_partitions,density = False)
		plt.title("Number of Partitions at epsilon = " + str(eps) + " Histogram: " + name)
		plt.show()

	def numbers_into_percents(self, attributes):
		'''
		Given a list of attribute names that hold numerical values, adds in atrributes 
		to the graph that normalize these attributes into decimals (relative to each other)
		Assumes the attribute names are strings
		'''
		new_attributes = []
		dict_list = []
		for attribute in attributes:
			new_attributes.append(attribute + "_PCT")
			dict_list.append(dict())
		for node in self.graph.nodes:
			node_total = sum([self.graph.nodes[node][attribute] for attribute in attributes])
			for i in range(len(attributes)):
				if node_total != 0:
					dict_list[i][node] = self.graph.nodes[node][attributes[i]]/node_total
				else:
					dict_list[i][node] = 0
		for i in range(len(attributes)):
			nx.set_node_attributes(self.graph, dict_list[i], new_attributes[i]) 

	def rand_tree_partition(self, k):
		edges = random.sample(self.tree.edges(), k-1)
		return trees.partition_subgraphs(self.tree,edges)

	def rand_partition(self, k, eps):
		balanced = False
		while not balanced:
			tree = get_spanning_tree(self.graph, self.node_communities)
			partitions = trees.balanced_population_partition(tree,k,eps)
			if len(partitions) > 0:
				balanced = True
				edges = random.choice(list(partitions))
		return trees.partition_subgraphs(tree,edges)

class Districting_Map:

	def __init__(self,graph, node_id_name, geo_df, df_id_col = None):
		self.districting_graph = Districting_Graph(graph)
		self.df = geo_df

		self.df["partition"] = 0
		self.partition_graph = nx.Graph()
		self.partition_graph.add_nodes_from([0])

		self.graph_id = node_id_name
		if df_id_col:
			self.df_id = df_id_col
		else:
			self.df_id = node_id_name

	def generate_rand_graph_tree(self):
		self.districting_graph.generate_rand_tree()

	def get_graph(self):
		return self.districting_graph.graph

	def get_graph_tree(self):
		return self.districting_graph.tree

	def set_partition_graph(self):
		partitions = self.get_partition_node_dict()
		#print(partitions)
		partition_graph = nx.Graph()
		partition_graph.add_nodes_from(self.df["partition"].unique())
		for part_tuple in itertools.combinations(partitions,2):
			partition_i = part_tuple[0]
			partition_j = part_tuple[1]
			nodes_partition_i = partitions[partition_i]
			nodes_partition_j = partitions[partition_j]
			for potential_edge in itertools.product(nodes_partition_i,nodes_partition_j):
				if self.districting_graph.graph.has_edge(*potential_edge):
					partition_graph.add_edge(partition_i,partition_j)
					break
		#print(partition_graph.edges)
		self.partition_graph = partition_graph

	def set_partitions(self, partitions):
		i = 0
		for partition in partitions:
			self.df.loc[self.df[self.df_id].isin(partitions[i]), "partition"] = i
			i += 1
		self.set_partition_graph()

	def edit_partitions(self, partitions_to_edit, partition_nodes):
		edits = len(partitions_to_edit)
		for edit in range(edits):
			partition = partitions_to_edit[edit]
			nodes = partition_nodes[edit]
			self.df.loc[self.df[self.df_id].isin(nodes), "partition"] = partition

	def set_partition_to_column(self, columns_name):
		self.df["partition"] = self.df[columns_name]
		partitions = self.get_partition_node_dict()
		#print(partitions)
		partition_graph = nx.Graph()
		partition_graph.add_nodes_from(self.df["partition"].unique())
		for part_tuple in itertools.combinations(partitions,2):
			partition_i = part_tuple[0]
			partition_j = part_tuple[1]
			nodes_partition_i = partitions[partition_i]
			nodes_partition_j = partitions[partition_j]
			for potential_edge in itertools.product(nodes_partition_i,nodes_partition_j):
				if self.districting_graph.graph.has_edge(*potential_edge):
					partition_graph.add_edge(partition_i,partition_j)
					break
		self.partition_graph = partition_graph

	def get_partition_node_list(self):
		parts = self.df["partition"].unique()
		partitions = []
		for part in parts:
			part_df = self.df[self.df["partition"] == part]
			partitions.append(frozenset(part_df[self.df_id]))
		return partitions

	def get_partition_node_dict(self):
		parts = self.df["partition"].unique()
		partitions = dict()
		for part in parts:
			part_df = self.df[self.df["partition"] == part]
			partitions[part] = frozenset(part_df[self.df_id])
		return partitions

	def get_rand_partition(self,k,eps):
		partitions = self.districting_graph.rand_partition(k, eps)
		return partitions

	def set_rand_partition(self, k, eps):
		partitions = self.districting_graph.rand_partition(k, eps)
		#print(partitions)
		i = 0
		for partition in partitions:
			self.df.loc[self.df['ID'].isin(partitions[i]), 'partition'] = i
			i += 1
		self.set_partition_graph()

	def plot_districting(self, colors = "tab20", legend = True):
		self.df.plot(column="partition",cmap = colors, legend = legend)
		plt.show()

	def plot_districts_by_demographic(self, attribute, colors = "Reds", legend = True):
		self.df["Plot_col"] = 0
		graph = self.districting_graph.graph
		partitions = self.get_partition_node_list()
		for partition in partitions:
			partition_population = 0
			partition_attribute = 0
			for node in partition:
				partition_population += graph.nodes[node]["population"]
				partition_attribute += graph.nodes[node][attribute]*graph.nodes[node]["population"]
			self.df.loc[self.df['ID'].isin(partition), 'Plot_col'] = partition_attribute/partition_population

		self.df.plot(column = "Plot_col", cmap = colors, legend = legend)
		plt.show()
		self.df.drop(["Plot_col"],axis = 1)

	def add_attribute_percentages(self, attributes):
		self.districting_graph.numbers_into_percents(attributes)

	def district_demographic_breakdown(self, attribute):
		graph = self.districting_graph.graph
		partitions = self.get_partition_node_dict()
		demo_breakdown = dict()
		for partition in partitions:
			partition_population = 0
			partition_attribute = 0
			nodes = partitions[partition]
			for node in nodes:
				partition_population += graph.nodes[node]["population"]
				partition_attribute += graph.nodes[node][attribute]*graph.nodes[node]["population"]
			demo_breakdown[partition] = partition_attribute/partition_population
		return demo_breakdown

	def evaluate_on_metric(self,metric):
		return metric(self.get_graph(),self.get_partition_node_list())

	def optimize_global(self, functions, coeffs, epsilon, k, samples, use_current_partition = False, view_scores = False):
		districting_cost = Linear_Districting_Cost(functions, coeffs, epsilon)
		districting_cost.global_optimizer(self, k , samples, use_current_partition, view_scores)

	def optimize_local(self, functions, coeffs, epsilon, iterations, samples_per_iteration, number_districts = 2, 
						use_current_partition = True, view_scores = False, verbose = True, track_progress = True):
		districting_cost = Linear_Districting_Cost(functions, coeffs, epsilon)
		districting_cost.pairwise_optimizer(self, iterations, samples_per_iteration, number_districts, 
							use_current_partition, view_scores, verbose, track_progress)

	def outlier_analysis_global_iteration(self,metric,samples,epsilon,track_progress = False):
		oa = Districting_Outlier_Analysis(metric)
		oa.outlier_analysis_globally_iterated(self, samples, epsilon, track_progress)


	def outlier_analysis_local_iteration(self,metric):
		oa = Districting_Outlier_Analysis(metric)






