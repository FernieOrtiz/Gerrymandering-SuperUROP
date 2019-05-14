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

class Linear_Districting_Cost():

	def __init__(self, functions, coeffs, population_eps = .01):
		self.functions = functions
		self.coeffs = coeffs
		self.eps = population_eps

	def evaluate(self,graph,partitions, view_scores = False):
		costs = []
		for i in range(len(self.functions)):
			cost_contribution = self.coeffs[i] * self.functions[i](graph,partitions)
			costs.append(cost_contribution)
		if view_scores:
			print("Sub-Scores: " ,costs)
		return sum(costs)

	def global_optimizer(self, districting_map, k, tree_samples, use_current_partition = False, view_scores = False):
		opt_score = np.inf
		opt_partition = None
		if use_current_partition:
			curr_partitions = districting_map.get_partition_node_list()
			opt_score = self.evaluate(districting_map.get_graph(), curr_partitions)
			opt_partition = curr_partitions
		for i in range(tree_samples):
			print("Sample " + str(i))
			districting_map.generate_rand_graph_tree()
			tree = districting_map.get_graph_tree()
			part_edges = list(trees.balanced_population_partition(tree,k,self.eps))
			for edge_tuple in part_edges:
			#for edge_tuple in itertools.combinations(tree.edges, k-1):
				partitions = trees.partition_subgraphs(tree, edge_tuple)
				score = self.evaluate(districting_map.get_graph(), partitions)
				if score < opt_score:
					opt_partition = partitions
					print("Improvement")
					opt_score = score
		print("Best Score Found: " + str(opt_score))
		districting_map.set_partitions(opt_partition)

	def pairwise_optimizer(self, districting_map, iterations, samples_per_iteration, number_districts = 2, 
							use_current_partition = False, view_scores = False, verbose = True, track_progress = True):

		k = len(districting_map.df["partition"].unique())
		graph = districting_map.get_graph()
		e = graph.number_of_edges()
		total_population = sum([graph.nodes[node]["population"] for node in graph.nodes])
		target_population = total_population/k

		if use_current_partition:
			init_partitions = districting_map.get_partition_node_list()
			opt_score = self.evaluate(graph, init_partitions, view_scores)
		else:
			districting_map.set_rand_partition(k)
			init_partitions = districting_map.get_partition_node_list()
			opt_score = self.evaluate(graph, init_partitions)

		partition_graph = districting_map.partition_graph
		curr_global_score = opt_score
		if verbose:
			print("Initial Score: " + str(curr_global_score))
		original_graph_edges = partition_graph.edges
		district_selection_sequence = []
		if verbose:
			print(partition_graph.edges)


		for iteration in range(iterations):

			neighbor_districts = [random.choice(list(districting_map.partition_graph.edges))]
			remaining_district_picks = number_districts - 2
			while remaining_district_picks > 0:
				selected_districts = set()
				for neighbors in neighbor_districts:
					selected_districts.add(neighbors[0])
					selected_districts.add(neighbors[1])
				neighbors_of_selected = list(districting_map.partition_graph.edges(selected_districts))
				neighbors_of_selected = [edge for edge in neighbors_of_selected if not trees.edge_in(edge,neighbor_districts)]
				neighbor_districts.append(random.choice(list(neighbors_of_selected)))
				remaining_district_picks -= 1

			district_selection_sequence.append(neighbor_districts)
			if verbose:
				print(neighbor_districts)
			partition_ids = set()
			for neighbors in neighbor_districts:
				partition_ids.add(neighbors[0])
				partition_ids.add(neighbors[1])
			partition_ids = list(partition_ids)
			partition_nodes = []
			for part in partition_ids:
				part_nodes = set(districting_map.df.loc[districting_map.df["partition"]==part,districting_map.df_id])
				partition_nodes.append(part_nodes)
			sub_nodes = set()
			for nodes in partition_nodes:
				sub_nodes = sub_nodes | nodes
			
			subgraph = graph.subgraph(sub_nodes)
			print(sum([subgraph.nodes[node]["population"] for node in subgraph.nodes]))
			districting_subgraph = Districting_Graph(subgraph)
			sub_score = self.evaluate(subgraph, partition_nodes)
			print(sub_score)
			opt_sub_score = sub_score
			opt_sub_partition = partition_nodes

			for sample in range(samples_per_iteration):
				if track_progress:
					print("Iteration " + str(iteration) + ", Sample " + str(sample))
				districting_subgraph.generate_rand_tree()
				tree = districting_subgraph.tree
				part_edges = list(trees.balanced_population_partition(tree,number_districts,self.eps, target = target_population))
				#print(part_edges)
				for edge_tuple in part_edges:
					partitions = trees.partition_subgraphs(tree, edge_tuple)
					curr_subscore = self.evaluate(districting_subgraph.graph, partitions)
					#print(curr_subscore)
					if curr_subscore < opt_sub_score:
						if track_progress:
							print("Improvement")
						opt_sub_partition = partitions
						opt_sub_score = curr_subscore

			districting_map.edit_partitions(partition_ids,opt_sub_partition)
			districting_map.set_partition_graph()
			#districting_map.plot_districting()
		if verbose:
			print("Sequence",district_selection_sequence)
		if view_scores:
			init_partitions = districting_map.get_partition_node_list()
			curr_score = self.evaluate(graph, init_partitions, view_scores)
		#print(original_graph_edges, partition_graph.edges)


