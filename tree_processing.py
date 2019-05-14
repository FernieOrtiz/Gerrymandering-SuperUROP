import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np
import make_graph 
import itertools
from networkx.algorithms import tree
import pprint
import pickle

def edge_equality(edge1, edge2):
	return ((edge1[0] == edge2[0]) and (edge1[1] == edge2[1])) or ((edge1[0] == edge2[1]) and (edge1[1] == edge2[0]))

def edge_in(edge, edge_collection):
	return (edge in edge_collection) or ((edge[1],edge[0]) in edge_collection)

def within_epsilon(p,target,epsilon):
	return (p >= (1-epsilon)*target and p <= (1+epsilon)*target)

def standardize_edges(edges):
	result =[]
	for edge in edges:
		if edge[0] < edge[1]:
			result.append(edge)
		else:
			result.append((edge[1],edge[0]))
	return result

def leaf_partition(graph, pop_col, pop_target, epsilon): 
	'''
	This function returns the edges of a graph that can be cut such that (at least) one of the 
	resulting components of the graph has a population within epsilon of the target population.

	Inputs:
		graph: A networkx graph where each vertex/node has a population attribute
		pop_col: String of the name of the graph's population attribute
		pop_target: Integer, the desired target population.
		epsilon: allowed deviation from target population
	Output:
		List of edges that result in one of the components is epsilon-close to target
		population.
	'''
	queue = get_leaves(graph)[:-1]
	random.shuffle(queue)
	explored = set()
	part_edges = []
	parent_dict = {}
	while len(queue)>1:
		node = queue.pop()
		if node not in explored:
			explored.add(node)
			out_edges = graph.edges(node)
			selected_edge = [edge for edge in out_edges if edge[1] not in explored][0]
			neighbor = selected_edge[1]
			if node in parent_dict:
				edge_pop = graph.nodes[node][pop_col] + parent_dict[node]
				if  edge_pop >= (1-epsilon)*pop_target and edge_pop <= (1+epsilon)*pop_target:
					part_edges.append(selected_edge)
			else:
				edge_pop = graph.nodes[node][pop_col]
				if  edge_pop >= (1-epsilon)*pop_target and edge_pop <= (1+epsilon)*pop_target:
					part_edges.append(selected_edge)
			if neighbor in parent_dict:
				parent_dict[neighbor] += edge_pop
			else:
				parent_dict[neighbor] = edge_pop
			neighbor_out_edges = graph.edges(neighbor)
			if len([edge for edge in neighbor_out_edges if edge[1] not in explored]) == 1:
				queue.append(neighbor)
	return part_edges

def augmented_leaf_partition(graph, total_population, pop_target, epsilon): 
	'''
	This function returns the edges of a graph that can be cut such that (at least) one of the 
	resulting components of the graph has a population within epsilon of the target population.

	Inputs:
		graph: A networkx graph where each vertex/node has a population attribute
		pop_col: String of the name of the graph's population attribute
		pop_target: Integer, the desired target population.
		epsilon: allowed deviation from target population
	Output:
		List of edges that result in one of the components is epsilon-close to target
		population.
	'''
	queue = get_leaves(graph)
	random.shuffle(queue)
	explored = set()
	part_edges = []
	parent_dict = {}
	while len(queue)>1:
		node = queue.pop()
		if node not in explored:
			explored.add(node)
			out_edges = graph.edges(node)
			selected_edge = [edge for edge in out_edges if edge[1] not in explored][0]
			neighbor = selected_edge[1]
			if node in parent_dict:
				edge_pop = graph.nodes[node]["population"] + parent_dict[node]
				if (within_epsilon(edge_pop, pop_target, epsilon) or within_epsilon(total_population - edge_pop, pop_target, epsilon)) and graph.edges[selected_edge]["cut"]:
					part_edges.append((selected_edge, edge_pop, total_population - edge_pop))
			else:
				edge_pop = graph.nodes[node]["population"]
				if (within_epsilon(edge_pop, pop_target, epsilon) or within_epsilon(total_population - edge_pop, pop_target, epsilon)) and graph.edges[selected_edge]["cut"]:
					part_edges.append((selected_edge, edge_pop, total_population - edge_pop))
			if neighbor in parent_dict:
				parent_dict[neighbor] += edge_pop
			else:
				parent_dict[neighbor] = edge_pop
			neighbor_out_edges = graph.edges(neighbor)
			if len([edge for edge in neighbor_out_edges if edge[1] not in explored]) == 1:
				queue.append(neighbor)
	return part_edges

def get_leaves(graph):
	"""
	Returns a list of the nodes of graph that are leaves
	"""
	return [x[0] for x in graph.degree if x[1] == 1]

def DFS_iterative_exploration(graph, start_node, blocked_edges = []):
	total_population = 0
	explored, stack = set(), [start_node]
	while stack:
		node = stack.pop()
		if node not in explored:
			explored.add(node)
			stack += [edge[1] for edge in graph.edges(node) if (edge[1] not in explored and not edge_in(edge,blocked_edges))]
			total_population += graph.nodes[node]["population"]
	return explored, total_population

def balanced_population_partition(graph,k,epsilon, target = None):
	total_population = sum([graph.nodes[node]["population"] for node in graph.nodes])
	if not target:
		target = total_population/k
	cut_edges = balanced_population_partition_method(graph, total_population, target, k, epsilon)
	edges = set()
	for edge_list in cut_edges:
		edge_set = frozenset(tuple(standardize_edges(edge_list)))
		edges.add(edge_set)
	return edges

def balanced_population_partition_method(graph, total_population, target, k, epsilon, blocked_edges = []):
	edges_returned = []
	if k == 2:
		edge_pop_tuples = augmented_leaf_partition(graph, total_population, target, epsilon)
		return [[edge_pop_tuple[0]] for edge_pop_tuple in edge_pop_tuples if 
				(within_epsilon(edge_pop_tuple[1],target,epsilon) and within_epsilon(edge_pop_tuple[2],target,epsilon))]
	else:
		edge_pop_tuples = augmented_leaf_partition(graph, total_population, target, epsilon)
		for edge_pop_tuple in edge_pop_tuples:
			edge = edge_pop_tuple[0]
			up_population = edge_pop_tuple[1]
			down_population = edge_pop_tuple[2]
			if within_epsilon(up_population,target,epsilon):
				next_start_node = edge[1]
				next_blocked_edges = blocked_edges + [edge]
				sub_nodes, sub_population = DFS_iterative_exploration(graph, next_start_node, blocked_edges = next_blocked_edges)
				subgraph = graph.subgraph(sub_nodes)
				recur_result = balanced_population_partition_method(subgraph, sub_population, target, k-1, epsilon, next_blocked_edges)
				for found_edge in recur_result:
					edges_returned.append(found_edge+[edge])
			if within_epsilon(down_population,target,epsilon):
				next_start_node = edge[0]
				next_blocked_edges = blocked_edges + [edge]
				sub_nodes, sub_population = DFS_iterative_exploration(graph, next_start_node, blocked_edges = next_blocked_edges)
				subgraph = graph.subgraph(sub_nodes)
				recur_result = balanced_population_partition_method(subgraph, sub_population, target, k-1, epsilon, next_blocked_edges)
				for found_edge in recur_result:
					edges_returned.append(found_edge+[edge])
	return edges_returned

def balanced_population_partition_exists(graph, k, epsilon):
	total_population = sum([graph.nodes[node]["population"] for node in graph.nodes])
	return balanced_population_partition_exists_method(graph,total_population,total_population/k,k,epsilon)

def balanced_population_partition_exists_method(graph, total_population, target, k, epsilon, blocked_edges = []):
	edges_returned = []
	if k == 2:
		edge_pop_tuples = augmented_leaf_partition(graph, total_population, target, epsilon)
		for edge_pop_tuple in edge_pop_tuples:
			if (within_epsilon(edge_pop_tuple[1],target,epsilon) and within_epsilon(edge_pop_tuple[2],target,epsilon)):
				return True
		return False
	else:
		edge_pop_tuples = augmented_leaf_partition(graph, total_population, target, epsilon)
		for edge_pop_tuple in edge_pop_tuples:
			edge = edge_pop_tuple[0]
			up_population = edge_pop_tuple[1]
			down_population = edge_pop_tuple[2]
			if within_epsilon(up_population,target,epsilon):
				next_start_node = edge[1]
				next_blocked_edges = blocked_edges + [edge]
				sub_nodes, sub_population = DFS_iterative_exploration(graph, next_start_node, blocked_edges = next_blocked_edges)
				subgraph = graph.subgraph(sub_nodes)
				recur_result = balanced_population_partition_exists_method(subgraph, sub_population, target, k-1, epsilon, next_blocked_edges)
				if recur_result:
					return True
			if within_epsilon(down_population,target,epsilon):
				next_start_node = edge[0]
				next_blocked_edges = blocked_edges + [edge]
				sub_nodes, sub_population = DFS_iterative_exploration(graph, next_start_node, blocked_edges = next_blocked_edges)
				subgraph = graph.subgraph(sub_nodes)
				recur_result = balanced_population_partition_exists_method(subgraph, sub_population, target, k-1, epsilon, next_blocked_edges)
				if recur_result:
					return True
		return False


def add_edge_pops_tree(graph):
	'''
	Adds the following attributes to each edge in the graph:
	left_node - The node to the "left" of the edge
	left_nodes - The set of nodes that lie to the "left" of the edge
	left_pop - The combines population of all nodes to the "left" of the edge
	right_node - The node to the "right" of the edge
	right_nodes - The set of nodes that lie to the "right" of the edge
	right_pop - The combines population of all nodes to the "right" of the edge

	*** Note: This function assumes that pop_col of the graph is "population"
	'''
	total_population = sum([graph.nodes[node]["population"] for node in graph.nodes])
	left_node_dict = dict()
	left_pop_dict = dict()
	right_node_dict = dict()
	right_pop_dict = dict()
	left_nodes_dict = dict()
	right_nodes_dict = dict()
	node_set = set(graph.nodes)
	queue = get_leaves(graph)
	random.shuffle(queue)
	explored = set()
	parent_dict = {}
	parent_edge_dict = {}
	#print(nx.is_connected(graph),nx.is_tree(graph))
	while len(queue)>1:
		node = queue.pop()
		if node not in explored:
			explored.add(node)
			out_edges = graph.edges(node)
			selected_edge = [edge for edge in out_edges if edge[1] not in explored][0]
			neighbor = selected_edge[1]
			if node in parent_dict:
				edge_pop = graph.nodes[node]["population"] + parent_dict[node]
				left_nodes = parent_edge_dict[node] | {node}
				left_node_dict[selected_edge]= node
				left_pop_dict[selected_edge] = edge_pop
				right_node_dict[selected_edge] = neighbor
				right_pop_dict[selected_edge] = total_population - edge_pop
				left_nodes_dict[selected_edge]= left_nodes
				right_nodes_dict[selected_edge] = node_set - left_nodes
			else:
				edge_pop = graph.nodes[node]["population"]
				left_nodes = {node}
				left_node_dict[selected_edge] = node
				left_pop_dict[selected_edge] = edge_pop
				right_node_dict[selected_edge] = neighbor
				right_pop_dict[selected_edge] = total_population - edge_pop
				left_nodes_dict[selected_edge] = left_nodes
				right_nodes_dict[selected_edge] = node_set - {node}

			if neighbor in parent_dict:
				parent_dict[neighbor] += edge_pop
				parent_edge_dict[neighbor] = parent_edge_dict[neighbor] | left_nodes
			else:
				parent_dict[neighbor] = edge_pop
				parent_edge_dict[neighbor] = left_nodes
			neighbor_out_edges = graph.edges(neighbor)
			if len([edge for edge in neighbor_out_edges if edge[1] not in explored]) == 1:
				queue.append(neighbor)
		'''
		except IndexError:
			out_edges = graph.edges(node)
			print(graph.edges(node))
			print([edge for edge in out_edges if edge[1] not in explored])
		'''

	nx.set_edge_attributes(graph,left_node_dict,'left_node')
	nx.set_edge_attributes(graph,left_pop_dict,'left_pop')
	nx.set_edge_attributes(graph,right_node_dict,'right_node')
	nx.set_edge_attributes(graph,right_pop_dict,'right_pop')
	nx.set_edge_attributes(graph,left_nodes_dict,'left_nodes')
	nx.set_edge_attributes(graph,right_nodes_dict,'right_nodes')
	return graph

def processed_balanced_population_partition(graph,k,epsilon, target = None):

	total_population = sum([graph.nodes[node]["population"] for node in graph.nodes])
	if not target:
		target = total_population/k
	cut_edges = population_partition(graph,target,k,epsilon)
	edges = set()
	for edge_list in cut_edges:
		edge_set = frozenset(tuple(edge_list))
		if edge_set not in edges:
			edges.add(frozenset(tuple(edge_list)))
	return edges

def processed_population_partition(graph,target,k,epsilon,cut_edge_dict = dict()):
	'''
	Returns the set of k-1 edges that can be cut resulting in each of the k connected components
	of the graph having total population within epsilon of target (usually total_population/k). 
	This code assumes the tree has been preprocessed by add_edge_pops_tree. 
	'''
	edges_returned = []
	curr_pop_dict = dict()
	node_dict = dict()
	if k == 2: #Base Case of this Recursive Program, 
		for edge in graph.edges:
			cut_edges_left = [cut_edge for cut_edge in cut_edge_dict if cut_edge[0] in graph.edges[edge]["left_nodes"]]
			cut_edges_right = [cut_edge for cut_edge in cut_edge_dict if cut_edge[0] in graph.edges[edge]["right_nodes"]]
			left_pop = graph.edges[edge]["left_pop"] - sum([cut_edge_dict[cut_edge] for cut_edge in cut_edges_left])
			right_pop = graph.edges[edge]["right_pop"] - sum([cut_edge_dict[cut_edge] for cut_edge in cut_edges_right])
			if (left_pop >= (1-epsilon)*target) & (left_pop <= (1+epsilon)*target) & (right_pop  >= (1-epsilon)*target) & (right_pop <= (1+epsilon)*target):
				edges_returned.append(edge)
		return [[edge] for edge in edges_returned]
	else: #Recursive Step 
		for edge in graph.edges:
			cut_edges_left = [cut_edge for cut_edge in cut_edge_dict if cut_edge[0] in graph.edges[edge]["left_nodes"]]
			cut_edges_right = [cut_edge for cut_edge in cut_edge_dict if cut_edge[0] in graph.edges[edge]["right_nodes"]]
			left_pop = graph.edges[edge]["left_pop"] - sum([cut_edge_dict[cut_edge] for cut_edge in cut_edges_left])
			right_pop = graph.edges[edge]["right_pop"] - sum([cut_edge_dict[cut_edge] for cut_edge in cut_edges_right])
			if (left_pop >= (1-epsilon)*target) & (left_pop <= (1+epsilon)*target):
				edges_returned.append(edge)
				curr_pop_dict[edge] = left_pop
				node_dict[edge] = graph.edges[edge]["right_nodes"] & set(graph.nodes)
			elif (right_pop  >= (1-epsilon)*target) & (right_pop <= (1+epsilon)*target):
				edges_returned.append(edge)
				curr_pop_dict[edge] = right_pop
				node_dict[edge] = graph.edges[edge]["left_nodes"] & set(graph.nodes)
		result = []
		for edge in edges_returned:
			new_edge_dict = dict(cut_edge_dict)
			new_edge_dict[edge] = curr_pop_dict[edge]
			subgraph = graph.subgraph(node_dict[edge])
			cuts = population_partition(subgraph,target,k-1,epsilon,new_edge_dict)
			for cut in cuts:
				cut.append(edge)
				result.append(cut)
		return result

def processed_balanced_population_exists(graph,k,epsilon):

	total_population = sum([graph.nodes[node]["population"] for node in graph.nodes])
	return  population_partition_exists(graph,total_population/k,k,epsilon)

def processed_partition_exists(graph,target,k,epsilon,cut_edge_dict = dict()):
	
	edges_returned = []
	curr_pop_dict = dict()
	node_dict = dict()
	if k == 2:
		for edge in graph.edges:
			cut_edges_left = [cut_edge for cut_edge in cut_edge_dict if cut_edge[0] in graph.edges[edge]["left_nodes"]]
			cut_edges_right = [cut_edge for cut_edge in cut_edge_dict if cut_edge[0] in graph.edges[edge]["right_nodes"]]
			left_pop = graph.edges[edge]["left_pop"] - sum([cut_edge_dict[cut_edge] for cut_edge in cut_edges_left])
			right_pop = graph.edges[edge]["right_pop"] - sum([cut_edge_dict[cut_edge] for cut_edge in cut_edges_right])
			if (left_pop >= (1-epsilon)*target) & (left_pop <= (1+epsilon)*target) & (right_pop  >= (1-epsilon)*target) & (right_pop <= (1+epsilon)*target):
				return True
		return False

	else:
		for edge in graph.edges:
			cut_edges_left = [cut_edge for cut_edge in cut_edge_dict if cut_edge[0] in graph.edges[edge]["left_nodes"]]
			cut_edges_right = [cut_edge for cut_edge in cut_edge_dict if cut_edge[0] in graph.edges[edge]["right_nodes"]]
			left_pop = graph.edges[edge]["left_pop"] - sum([cut_edge_dict[cut_edge] for cut_edge in cut_edges_left])
			right_pop = graph.edges[edge]["right_pop"] - sum([cut_edge_dict[cut_edge] for cut_edge in cut_edges_right])
			if (left_pop >= (1-epsilon)*target) & (left_pop <= (1+epsilon)*target):
				edges_returned.append(edge)
				curr_pop_dict[edge] = left_pop
				node_dict[edge] = graph.edges[edge]["right_nodes"] & set(graph.nodes)
			elif (right_pop  >= (1-epsilon)*target) & (right_pop <= (1+epsilon)*target):
				#print("Right Cut ", edge, ", Population ",right_pop)
				edges_returned.append(edge)
				curr_pop_dict[edge] = right_pop
				node_dict[edge] = graph.edges[edge]["left_nodes"] & set(graph.nodes)
		result = []
		for edge in edges_returned:
			new_edge_dict = dict(cut_edge_dict)
			new_edge_dict[edge] = curr_pop_dict[edge]
			subgraph = graph.subgraph(node_dict[edge])
			if population_partition_exists(subgraph,target,k-1,epsilon,new_edge_dict):
				return True
		return False


def dfs_balanced_population_partition(graph, k, epsilon):
	total_population = sum([graph.nodes[node]["population"] for node in graph.nodes])
	target = total_population/k
	start_node = random.choice(list(graph.nodes))
	cut_edges = population_partition_DFS(graph, start_node, target, total_population, k, epsilon)
	edges = set()
	for edge_list in cut_edges:
		edge_set = frozenset(tuple(standardize_edges(edge_list)))
		if edge_set not in edges:
			edges.add(frozenset(tuple(edge_list)))
	return edges

def population_partition_DFS(graph, start_node, target, curr_total_population, k, epsilon, blocked_edges = []):
	edges_returned = []
	if k == 2:
		edge_pop_tuples = DFS_explore_for_partition(graph, start_node, curr_total_population, blocked_edges = blocked_edges)
		return [[edge_pop_tuple[0]] for edge_pop_tuple in edge_pop_tuples if 
				(within_epsilon(edge_pop_tuple[1],target,epsilon) and within_epsilon(edge_pop_tuple[2],target,epsilon))]
	else:
		edge_pop_tuples = DFS_explore_for_partition(graph, start_node, curr_total_population, blocked_edges = blocked_edges)
		print(edge_pop_tuples)
		for edge_pop_tuple in edge_pop_tuples:
			edge = edge_pop_tuple[0]
			up_population = edge_pop_tuple[1]
			down_population = edge_pop_tuple[2]
			if within_epsilon(up_population,target,epsilon):
				next_start_node = edge[1]
				next_blocked_edges = blocked_edges + [edge]
				next_total_population = curr_total_population - up_population
				recur_result = population_partition_DFS(graph, next_start_node, target, next_total_population, k-1, epsilon, blocked_edges = next_blocked_edges)
				for found_edge in recur_result:
					edges_returned.append(found_edge+[edge])

			if within_epsilon(down_population,target,epsilon):
				next_start_node = edge[0]
				next_blocked_edges = blocked_edges + [edge]
				next_total_population = curr_total_population - down_population
				recur_result = population_partition_DFS(graph, next_start_node, target, next_total_population, k-1, epsilon, blocked_edges = next_blocked_edges)
				for found_edge in recur_result:
					edges_returned.append(found_edge+[edge])
	return edges_returned


def DFS_explore_for_partition(graph, node, total_population, parent = None, blocked_edges = []):
	returned_edges = []
	out_edges = graph.edges(node)
	neighbors = [edge[1] for edge in graph.edges(node) if (edge[1] != parent and not edge_in(edge,blocked_edges))]
	for neighbor in neighbors:
		if graph.degree[neighbor] == 1:
			returned_edges.append(((node,neighbor),total_population - graph.nodes[neighbor]["population"], graph.nodes[neighbor]["population"]))
		else:
			child_explore = DFS_explore_for_partition(graph, neighbor,total_population, node, blocked_edges)
			down_pop = graph.nodes[neighbor]["population"] + sum([edge_pop_pair[2] for edge_pop_pair in child_explore if neighbor in edge_pop_pair[0]])
			up_pop = total_population - down_pop
			returned_edges.append(((node,neighbor), up_pop, down_pop))
			returned_edges.extend(child_explore)
	return returned_edges

def partition_subgraphs(graph_tree,cut_edges):
	partitions = []
	edges = list(cut_edges)
	explored = set()
	for edge in edges:
		node1 = edge[0]
		node2 = edge[1] 
		if node1 not in explored:
			part = DFS_iterative_exploration(graph_tree, node1, blocked_edges = cut_edges)[0]
			explored.update(part)
			partitions.append(part)
		if node2 not in explored:
			part = DFS_iterative_exploration(graph_tree, node2, blocked_edges = cut_edges)[0]
			explored.update(part)
			partitions.append(part)
	return partitions


def processed_partition_subgraphs(graph_tree,cut_edges):
	partitions = []
	edges = list(cut_edges)
	for i in range(len(edges)):
		edge = edges[i]
		if i == 0:
			partitions.append(graph_tree.edges[edge]["left_nodes"])
			partitions.append(graph_tree.edges[edge]["right_nodes"])
		else:
			for j in range(len(partitions)):
				if edge[0] in partitions[j]:
					part = partitions[j]
					del partitions[j]
					partitions.append(graph_tree.edges[edge]["left_nodes"] & part)
					partitions.append(graph_tree.edges[edge]["right_nodes"] & part)
					break
	return partitions
