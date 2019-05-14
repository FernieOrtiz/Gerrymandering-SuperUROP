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

class Districting_Outlier_Analysis():

	def __init__(self, metric):
		self.metric = metric

	def outlier_analysis_globally_iterated(self, districting_map, samples, epsilon, track_progress):
		graph_score = districting_map.evaluate_on_metric(self.metric)
		k = len(districting_map.df["partition"].unique())
		scores = []
		iterations = 0
		while iterations < samples:
			districting_map.generate_rand_graph_tree()
			tree = districting_map.get_graph_tree()
			part_edges = list(trees.balanced_population_partition(tree,k,epsilon))
			for edge_tuple in part_edges:
				if iterations < samples:
					print("Sample: "+ str(iterations))
					partitions = trees.partition_subgraphs(tree, edge_tuple)
					scores.append(self.metric(districting_map.get_graph(),partitions))
					iterations += 1
		result = plt.hist(scores)
		plt.axvline(graph_score, color='k', linestyle='dashed', linewidth=1)
		plt.show()

	def outlier_analysis_locally_iterated(self, districting_map, samples, skipped_samples, number_districts = 2, 
							view_scores = False, verbose = True, track_progress = True):

		dm_copy = copy.deepcopy(districting_map)
		graph_score = dm_copy.evaluate_on_metric(self.metric)
		k = len(dm_copy.df["partition"].unique())
		graph = dm_copy.get_graph()
		e = graph.number_of_edges()
		total_population = sum([graph.nodes[node]["population"] for node in graph.nodes])
		target_population = total_population/k

		partition_graph = dm_copy.partition_graph
		curr_global_score = opt_score
		original_graph_edges = partition_graph.edges
		district_selection_sequence = []
		if verbose:
			print(partition_graph.edges)


		for iteration in range(iterations):

			neighbor_districts = [random.choice(list(dm_copy.partition_graph.edges))]
			remaining_district_picks = number_districts - 2
			while remaining_district_picks > 0:
				selected_districts = set()
				for neighbors in neighbor_districts:
					selected_districts.add(neighbors[0])
					selected_districts.add(neighbors[1])
				neighbors_of_selected = list(dm_copy.partition_graph.edges(selected_districts))
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
				part_nodes = set(dm_copy.df.loc[dm_copy.df["partition"]==part,dm_copy.df_id])
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
				for edge_tuple in part_edges:
					partitions = trees.partition_subgraphs(tree, edge_tuple)
					curr_subscore = self.evaluate(districting_subgraph.graph, partitions)
					if curr_subscore < opt_sub_score:
						if track_progress:
							print("Improvement")
						opt_sub_partition = partitions
						opt_sub_score = curr_subscore

			dm_copy.edit_partitions(partition_ids,opt_sub_partition)
			dm_copy.set_partition_graph()
		if verbose:
			print("Sequence",district_selection_sequence)
		if view_scores:
			init_partitions = districting_map.get_partition_node_list()
			curr_score = self.evaluate(graph, init_partitions, view_scores)

