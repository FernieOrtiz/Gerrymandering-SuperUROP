
def constant_score(graph,partitions):
	return 1

def num_outgoing_edges(graph, sub_nodes):
	subgraph = graph.subgraph(sub_nodes)
	internal_edges = subgraph.number_of_edges()
	#print(graph.degree(sub_nodes))
	degree_list = [node[1] for node in graph.degree(sub_nodes)]
	external_edges = sum(degree_list)- 2*internal_edges
	#print(external_edges)
	return external_edges

def outgoing_edge_score(graph,partitions):
	cross_edges = 0
	for partition in partitions:
		cross_edges += num_outgoing_edges(graph, partition)
	return cross_edges/(2*graph.number_of_edges())

def demographic_majority_districts_score(graph, partitions, attribute_name):
	majority_districts = 0
	for partition in partitions:
		partition_population = 0
		partition_attribute = 0
		for node in partition:
			partition_population += graph.nodes[node]["population"]
			partition_attribute += graph.nodes[node][attribute_name]*graph.nodes[node]["population"]
		if partition_attribute/partition_population > .5:
			majority_districts += 1
	return majority_districts

def district_packing_score(graph, partitions, attribute_name):
	score = 0
	for partition in partitions:
		partition_population = 0
		partition_attribute = 0
		for node in partition:
			partition_population += graph.nodes[node]["population"]
			partition_attribute += graph.nodes[node][attribute_name]*graph.nodes[node]["population"]
		if partition_attribute/partition_population > .5:
			score += 1 - partition_attribute/partition_population 
	return score

def efficiency_gap_score(graph, partitions, attribute_party1, attribute_party2):
	wasted_votes1 = 0
	wasted_votes2 = 0
	total_votes = 0
	for partition in partitions:
		party1_votes = 0
		party2_votes = 0
		nodes = partitions[partition]
		for node in nodes:
			party1_votes += graph.nodes[node][attribute_party1]
			party2_votes += graph.nodes[node][attribute_party2]
		threshhold = int((party1_votes + party2_votes) / 2) + 1
		if party1_votes > party2_votes:
			wasted_votes1 += party1_votes - threshhold
			wasted_votes2 += party2_votes
		else:
			wasted_votes2 += party2_votes - threshhold
			wasted_votes1 += party1_votes
		total_votes += party1_votes + party2_votes

	return (wasted_votes1 - wasted_votes2) / total_votes


def population_balance_score(graph, partitions, p, target = None):
	k = len(partitions)
	part_populations = []
	for partition in partitions:
		subgraph = graph.subgraph(partition)
		partition_population =  sum([subgraph.nodes[node]["population"] for node in subgraph.nodes])
		part_populations.append(partition_population)
	total_population = sum(part_populations)
	if not target:
		target = total_population/k
	deviations = np.array(part_populations)/target - 1
	return np.sum(np.abs(deviations)**p)