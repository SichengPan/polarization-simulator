import os
import json
import heapq
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import modularity, greedy_modularity_communities
from sklearn.decomposition import PCA


class Node:
    def __init__(self, node_id, opinion_dim=5, attribute_dim=0, attribute_vector=None):
        """
        Initialize a Node with an ID and multi-dimensional opinion vector.

        Parameters:
        - node_id: int, unique identifier for the node.
        - opinion_dim: int, number of dimensions for the opinion vector (default is 5).
        """
        self.node_id = node_id
        self.opinion = np.random.uniform(-1, 1, size=opinion_dim)  # Multi-dimensional opinion vector

        # use attribute_vector if provided, else generate a random one if specified, else do not use it
        if attribute_vector is not None:
            self.attribute_vector = np.array(attribute_vector)
        elif attribute_dim > 0:
            self.attribute_vector = np.random.uniform(0, 1, size=attribute_dim)
        else:
            self.attribute_vector = np.array([])

    def get_opinion(self):
        """
        Getter for the opinion vector.

        Returns:
        - numpy array: the current opinion vector of the node.
        """
        return self.opinion

    def set_opinion(self, new_opinion):
        """
        Setter for the opinion vector.

        Parameters:
        - new_opinion: numpy array, the new opinion vector for the node.
        """
        new_opinion = np.clip(new_opinion, -1, 1)
        self.opinion = np.array(new_opinion)

    def get_attributes(self):
        """
        Getter for the attribute vector.

        Returns:
        - numpy array: the current attribute vector of the node.
        """
        return self.attribute_vector

    def set_attributes(self, new_attributes):
        """
        Setter for the attribute vector.

        Parameters:
        - new_attributes: numpy array, the new attribute vector for the node.
        """
        self.attribute_vector = np.array(new_attributes)


class SocialNetworkModel:
    def __init__(self, N=50, p=0.1, gamma=0.9, K=1.0, alpha=1.0, _lambda=1, network_type="random",
                dynamic_edges=False, threshold_connect=0.75, threshold_disconnect=0.25, opinion_dim=5,
                seed=42, sample_size_connect=5, sample_size_drop=5, custom_network_func=None,
                neutral_ratio=0.0, pairwise_neutral=False, adaptive_neutral=False,
                neutral_opinion_setting="zero", custom_neutral_opinions=None, **kwargs):
        """
        Initialize the social network model with parameters and create nodes.
        Use 'erdos_renyi model (each pair of nodes has an independent, fixed probability p of being connected)' for sparse & dense networks initialization (random network with different connection probability)
        Use 'barabasi_albert model (each new node would have 3 neighbors, more likely to connect to those with more connections)' for scale-free networks initialization

        Parameters:
        - N: int, number of nodes in the network.
        - p: float, probability for edge creation in a random graph.
        - gamma: float, decay factor for opinions, controls the natural decline of an individual's opinion over time, in the absence of social influence.
          Formula: decay term = γ * x_i(t)
        - K: float, control parameter for social influence, determines the weight of social influence on each individual's opinion.
          Formula for social influence: (K / k_i) * Σ_j A_ij * tanh(α * (σ_i(t) * σ_j(t))^λ * x_j(t))
        - alpha: float, parameter controlling opinion translation into social influence.
          Modulates the intensity of opinion influence, affecting the input to the tanh function.
          Larger alpha values increase the opinion influence, making it reach its limit ±1 faster.
        - _lambda: 0 or 1, parameter for intergroup influence control.
          controls the backfire effect, where λ = 1 leads to opinion reinforcement due to out-group contacts.
          Formula for intergroup influence: tanh(α * (σ_i(t) * σ_j(t))^λ * x_j(t))
        """
        self.N = N
        self.gamma = gamma
        self.K = K
        self.alpha = alpha
        self._lambda = _lambda
        self.opinion_dim = opinion_dim
        self.network_type = network_type
        self.history = []

        self.dynamic_edges = dynamic_edges
        self.threshold_connect = threshold_connect
        self.threshold_disconnect = threshold_disconnect

        self.sample_size_connect = sample_size_connect
        self.sample_size_drop = sample_size_drop

        # Set the seed for reproducibility
        self.seed = seed
        np.random.seed(self.seed)  # Ensures opinion vectors are the same

        # Initialize the network based on the specified type
        if custom_network_func:
            self.graph = custom_network_func(N, **kwargs)
        elif network_type == "random":
            self.graph = nx.erdos_renyi_graph(N, p, seed=self.seed)  # Random graph using the Erdős–Rényi model
        elif network_type == "small_world":
            k = kwargs.get("k", 4)  # Each node is initially connected to `k` neighbors
            rewiring_prob = kwargs.get("rewiring_prob", 0.1)  # Probability of rewiring edges
            self.graph = nx.watts_strogatz_graph(N, k, rewiring_prob, seed=self.seed)
        elif network_type == "dense":
            self.graph = nx.complete_graph(N)  # Dense graph, represented as a complete graph
        elif network_type == "scale_free":
            m = kwargs.get("m", 3)  # each new node would have 3 neighbors, more likely to connect to those with more connections
            self.graph = nx.barabasi_albert_graph(N, m, seed=self.seed)  # Scale-free graph using the Barabási–Albert model
        else:
            raise ValueError("Unsupported network type: choose from 'random', 'dense', or 'scale_free'.")

        # Initialize nodes with random opinion vectors
        self.nodes = {i: Node(i, opinion_dim=self.opinion_dim) for i in range(N)}

        # Neutral Nodes
        self.neutral_opinion_setting = neutral_opinion_setting
        if neutral_ratio > 0:
            # Select a proportion of nodes as "neutral agents"
            self.num_neutral = int(N * neutral_ratio)
            self.neutral_nodes = np.random.choice(list(self.nodes.keys()), self.num_neutral, replace=False)

            # Manually set the opinion vectors for neutral agents
            self.custom_neutral_opinions = custom_neutral_opinions if custom_neutral_opinions else {}

            for node_id in self.neutral_nodes:
                if neutral_opinion_setting == "zero":
                    self.nodes[node_id].set_opinion(np.zeros(self.opinion_dim))
                elif neutral_opinion_setting == "random":
                    self.nodes[node_id].set_opinion(np.random.uniform(-0.3, 0.3, size=self.opinion_dim))
                elif neutral_opinion_setting == "custom" and node_id in self.custom_neutral_opinions:
                    self.nodes[node_id].set_opinion(np.array(self.custom_neutral_opinions[node_id]))

            # Invoke the pairwise neutral linking function
            self.pairwise_neutral = pairwise_neutral
            if self.pairwise_neutral:
                self.setup_pairwise_neutral_links()

            # Enable adaptive neutral agents if specified
            self.adaptive_neutral = adaptive_neutral
        else:
            self.num_neutral = 0
            self.neutral_nodes = []
            self.pairwise_neutral = False
            self.adaptive_neutral = False


    def initialize_random_attributes(self, attribute_dim=5):
        """
        Initializes each node with a random attribute vector of fixed length.

        Parameters:
        - attribute_dim (int): The length of the attribute vector.

        Effect:
        - Updates each node's `attribute_vector` with random values in the range [-1, 1].
        """

        for node in self.nodes.values():
            node.set_attributes(np.random.uniform(-1, 1, size=attribute_dim))

        print(f"✅ Initialized random attribute vectors with dimension {attribute_dim} for all nodes.")


    def convert_topk_to_neutral_nodes(self, top_k=1):
        """
        Identify the top-K highest-degree nodes in each community and convert them to neutral nodes.

        Parameters:
        - top_k (int): Number of highest-degree nodes in each community to be converted.
        """
        # Detect communities
        communities = list(greedy_modularity_communities(self.graph))

        # Identify top-k highest degree nodes in each community
        new_neutral_nodes = set()

        for community_index, community in enumerate(communities):
            if len(community) <= top_k:
                continue  # Skip small communities

            # Select top_k nodes by degree
            sorted_nodes = sorted(community, key=lambda node: self.graph.degree[node], reverse=True)
            core_nodes = sorted_nodes[:top_k]  # `top_k` highest-degree nodes

            # Add to neutral_nodes set
            new_neutral_nodes.update(core_nodes)

        # Convert the selected nodes to neutral nodes
        self.neutral_nodes = list(new_neutral_nodes)
        self.num_neutral = len(self.neutral_nodes)


        # Reset their opinions based on `neutral_opinion_setting`
        for node_id in self.neutral_nodes:
            if self.neutral_opinion_setting == "zero":
                self.nodes[node_id].set_opinion(np.zeros(self.opinion_dim))
            elif self.neutral_opinion_setting == "random":
                self.nodes[node_id].set_opinion(np.random.uniform(-0.3, 0.3, size=self.opinion_dim))
            elif self.neutral_opinion_setting == "custom" and node_id in self.custom_neutral_opinions:
                self.nodes[node_id].set_opinion(np.array(self.custom_neutral_opinions[node_id]))



    def setup_pairwise_neutral_links(self, min_connections=2, max_connections=3, top_k=2, reset_neutral_links=False):
        """
        Ensures that each neutral node connects to 2~3 different communities,
        and only connects to the highest-degree `top_k` nodes within each community.

        Parameters:
        - min_connections (int): Minimum number of connections each neutral node must have.
        - max_connections (int): Maximum number of connections each neutral node can have.
        - top_k (int): Number of highest-degree nodes in each community that can connect to neutral nodes.
        - reset_neutral_links (bool): If True, remove all existing connections of neutral nodes before relinking.
        """
        # (Optional) Remove all existing connections for neutral nodes
        if reset_neutral_links:
            print("🛠 Resetting all neutral node connections...")
            for neutral in self.neutral_nodes:
                self.graph.remove_edges_from(list(self.graph.edges(neutral)))

        # Detect communities
        communities = list(greedy_modularity_communities(self.graph))
        # np.random.seed(self.seed)

        # Identify the top-k highest degree nodes in each community
        community_core_nodes = {}
        for community_index, community in enumerate(communities):
            sorted_nodes = sorted(community, key=lambda node: self.graph.degree[node], reverse=True)
            core_nodes = sorted_nodes[:top_k]  # Select `top_k` core nodes

            # Skip communities with only neutral nodes
            if all(node in self.neutral_nodes for node in community):
                print(f"⚠️ Skipping Community {community_index} (Only Neutral Nodes)")
                continue

            community_core_nodes[community_index] = core_nodes

            # Debug: Print core nodes and their degrees
            print(f"📌 Community {community_index}:")
            for node in core_nodes:
                print(f"   ➤ Core Node {node} (Degree: {self.graph.degree[node]})")

        # Assign neutral nodes to 2~3 different communities
        for neutral in self.neutral_nodes:
            available_communities = list(community_core_nodes.keys())

            # Skip neutral node if no valid communities are available
            if not available_communities:
                print(f"⚠️ No valid communities for Neutral Node {neutral}, skipping...")
                continue

            # Select communities to connect to
            chosen_communities = np.random.choice(available_communities,
                                                  size=np.random.randint(min_connections, max_connections + 1),
                                                  replace=False)

            print(f"\n🔹 Assigning Neutral Node {neutral} to communities: {chosen_communities}")

            for community_index in chosen_communities:
                core_nodes = community_core_nodes[community_index]  # Get the core nodes of the selected community
                chosen_core = np.random.choice(core_nodes)  # Randomly select one core node
                self.graph.add_edge(neutral, chosen_core)  # Create a connection
                print(f"   ➤ Connected to Core Node {chosen_core} (Community {community_index}, Degree: {self.graph.degree[chosen_core]})")

        # Ensure all neutral nodes have at least `min_connections` edges
        for neutral in self.neutral_nodes:
            while self.graph.degree[neutral] < min_connections:
                available_communities = list(community_core_nodes.keys())

                if not available_communities:
                    print(f"⚠️ Neutral Node {neutral} has no available communities for additional connections!")
                    break

                additional_community = np.random.choice(available_communities)
                core_nodes = community_core_nodes[additional_community]
                chosen_core = np.random.choice(core_nodes)
                if not self.graph.has_edge(neutral, chosen_core):
                    self.graph.add_edge(neutral, chosen_core)
                    print(f"⚠️ Neutral Node {neutral} had insufficient connections! Added extra link to Core Node {chosen_core} (Community {additional_community})")


    def get_community_cores(self):
        """
        Return the list of core nodes from each community.
        """
        communities = list(greedy_modularity_communities(self.graph))
        core_nodes = [max(community, key=lambda node: self.graph.degree[node]) for community in communities]
        return set(core_nodes)


    def update_neutral_links(self):
        """
        If adaptive_neutral is enabled, periodically reassign neutral node connections to maintain effectiveness.
        """
        if not self.adaptive_neutral:
            return  # If adaptive behavior is disabled, do nothing

        print("🔄 Updating neutral node connections...")

        # Remove existing neutral node edges
        for neutral in self.neutral_nodes:
            neighbors = list(self.graph.neighbors(neutral))
            for neighbor in neighbors:
                if neighbor not in self.nodes:
                    continue
                # keep right links
                if neighbor not in self.get_community_cores():
                    self.graph.remove_edge(neutral, neighbor)

        # Reassign neutral nodes based on updated community structure
        self.setup_pairwise_neutral_links()

    """
    Rewrite the `propagate_opinions` method to include previous features, but easier to maintain
    """
    def compute_new_opinions(self):
        """
        Calculate the new opinions during the propagation.
        Following formula:
        - x_i(t + 1) = γ * x_i(t) + (K / k_i) * Σ_j A_ij * tanh(α * (σ_i(t) * σ_j(t))^λ * x_j(t))

        where:
        - γ (gamma): Decay factor, controlling the natural decline of opinions over time.
        - K: Control parameter for social influence, determining the weight of neighbors' influence.
        - α (alpha): Scaling factor for opinion influence, adjusting the intensity of social interactions.
        - λ (_lambda): Parameter controlling the effect of intergroup interactions.
        - A_ij: Adjacency matrix element (1 if nodes i and j are connected, otherwise 0).
        - σ_i(t): The sign of the opinion vector of node i.
        - σ_j(t): The sign of the opinion vector of node j.
        - x_j(t): The opinion vector of node j.
        - k_i: Degree of node i, normalizing social influence.

        Returns:
        - dict: A dictionary mapping node IDs to their new opinion vectors after propagation.
        """
        new_opinions = {}
        for node_id, node in self.nodes.items():
            # Skip neutral nodes - they should not update their opinions
            if node_id in self.neutral_nodes:
                new_opinions[node_id] = node.get_opinion()
                continue

            # Decay term: γ * x_i(t)
            decay_term = self.gamma * node.get_opinion()

            # Social influence term
            social_influence = np.zeros_like(decay_term)
            for neighbor_id in self.graph.neighbors(node_id):
                neighbor = self.nodes[neighbor_id]

                # Influence calculation: tanh(α * (σ_i(t) * σ_j(t))^λ * x_j(t))
                influence = np.tanh(
                    self.alpha
                    * ((np.sign(node.get_opinion()) * np.sign(neighbor.get_opinion())) ** self._lambda)
                    * neighbor.get_opinion()
                )
                social_influence += influence

            # (K / k_i) * Σ_j A_ij * tanh(α * (σ_i(t) * σ_j(t))^λ * x_j(t))
            if self.graph.degree[node_id] > 0:
                social_influence *= self.K / self.graph.degree[node_id]

            # γ * x_i(t) + (K / k_i) * Σ_j A_ij * tanh(α * (σ_i(t) * σ_j(t))^λ * x_j(t))
            new_opinion = decay_term + social_influence

            # update new opinions
            new_opinions[node_id] = np.clip(new_opinion, -1, 1)

        return new_opinions


    def update_opinions(self, new_opinions):
        """
        Apply computed opinion updates to the network nodes.

        This function updates each node's opinion based on the computed values and 
        records the network state after applying the updates.

        Arg:
            new_opinions (dict): A dictionary containing updated opinion values for each node.

        Returns:
            None
        """
        for node_id, new_opinion in new_opinions.items():
            self.nodes[node_id].set_opinion(new_opinion)

        # Record the current state after opinion propagation
        self.save_network_state()


    def propagate_opinions(self):
        """
        Update node opinions in the network using social influence and opinion decay.

        This method separates computation from application:
        1. `compute_new_opinions` computes updated opinion values.
        2. `apply_opinions` applies these values and updates the network state.
        """
        new_opinions = self.compute_new_opinions()
        self.update_opinions(new_opinions)

    """
    Rewrite the `adjust_edges_based_on_similarity` method to include previous features, but easier to maintain
    """
    def get_recommended_nodes(self, node_id, potential_new_connections,
                                use_hybrid_recommendation=False,
                                use_attribute_and_dissimilar_recommendation=False,
                                use_random_recommendation=False, 
                                use_content_based_recommendation=False,
                                use_dissimilar_recommendation=False, 
                                use_collaborative_recommendation=False,
                                use_attribute_vector_recommendation=False,
                                alpha_recommendation=1.0,
                                sample_size_hybrid=5,
                                sample_size_attribute_dissimilar=5, 
                                sample_size_random=5, 
                                sample_size_content=5, 
                                sample_size_dissimilar=5, 
                                sample_size_collaborative=5,
                                sample_size_attribute_vector=5):
        """
        Generate a set of recommended nodes for a given node using various recommendation algorithms.

        Args:
            node_id (int): The node for which recommendations are generated.
            potential_new_connections (set): Set of nodes not currently connected.
            use_random (bool): Whether to include random recommendations.
            use_content (bool): Whether to use content-based recommendations.
            use_dissimilar (bool): Whether to use dissimilarity-based recommendations.
            use_collaborative (bool): Whether to use collaborative filtering.
            alpha_recommendation (float): Scaling factor for recommendation-based similarity calculation.
            sample_size_random (int): Number of random nodes to consider.
            sample_size_content (int): Number of content-based nodes to consider.
            sample_size_dissimilar (int): Number of dissimilar nodes to consider.
            sample_size_collaborative (int): Number of collaborative filtering nodes to consider.

        Returns:
            set: A set of recommended node IDs.
        """
        recommended_nodes = set()

        if use_hybrid_recommendation and len(potential_new_connections) > 0:
            hybrid_nodes = self.recommend_hybrid(node_id, sample_size_hybrid, alpha_recommendation)
            recommended_nodes.update(hybrid_nodes)

        if use_attribute_and_dissimilar_recommendation and len(potential_new_connections) > 0:
                attribute_dissimilar_nodes = self.attribute_and_dissimilar_recommendation(node_id, sample_size_attribute_dissimilar, alpha_recommendation)
                recommended_nodes.update(attribute_dissimilar_nodes)

        if use_attribute_vector_recommendation and len(potential_new_connections) > 0:
            attribute_vector_nodes = self.recommend_by_attribute_vector(node_id, sample_size_attribute_vector)
            recommended_nodes.update(attribute_vector_nodes)

        if use_random_recommendation and len(potential_new_connections) > 0:
            random_nodes = self.random_recommendation(node_id, potential_new_connections, sample_size_random)
            recommended_nodes.update(random_nodes)

        if use_content_based_recommendation and len(potential_new_connections) > 0:
            content_based_nodes = self.content_based_recommendation(node_id, sample_size_content)
            recommended_nodes.update(content_based_nodes)

        if use_dissimilar_recommendation and len(potential_new_connections) > 0:
            dissimilar_nodes = self.dissimilar_recommendation(node_id, sample_size_dissimilar)
            recommended_nodes.update(dissimilar_nodes)

        if use_collaborative_recommendation and len(potential_new_connections) > 0:
            collaborative_nodes = self.collaborative_filtering_recommendation(node_id, sample_size_collaborative)
            recommended_nodes.update(collaborative_nodes)

        return recommended_nodes


    def evaluate_new_connections(self, node_id, recommended_nodes, threshold_connect, alpha_user):
        """
        Evaluate potential new connections and determine which should be added.

        Parameters:
        - node_id: int, the node for which connections are being evaluated.
        - recommended_nodes: set, recommended nodes for potential new connections.
        - threshold_connect: float, similarity threshold for adding a new edge.
        - alpha_user: float, parameter controlling similarity calculations.

        Returns:
        - list: Pairs of nodes representing edges to be added.
        """
        edges_to_add = []
        for other_node_id in recommended_nodes:
            similarity = self.calculate_similarity(node_id, other_node_id, alpha_user)
            if similarity >= threshold_connect:
                edges_to_add.append((node_id, other_node_id))
        return edges_to_add


    def evaluate_existing_connections(self, node_id, existing_connections, threshold_disconnect, alpha_user):
        """
        Evaluate existing connections and determine which should be removed.

        Parameters:
        - node_id: int, the node for which connections are being evaluated.
        - existing_connections: list, currently connected nodes.
        - threshold_disconnect: float, similarity threshold for removing an edge.
        - alpha_user: float, parameter controlling similarity calculations.

        Returns:
        - list: Pairs of nodes representing edges to be removed.
        """
        edges_to_remove = []
        for other_node_id in existing_connections:
            similarity = self.calculate_similarity(node_id, other_node_id, alpha_user)
            if similarity < threshold_disconnect and node_id not in self.neutral_nodes and other_node_id not in self.neutral_nodes:
                edges_to_remove.append((node_id, other_node_id))
        return edges_to_remove
    

    def apply_network_adjustments(self, edges_to_add, edges_to_remove):
        """
        Adjusts network structure based on its type before applying edge modifications.

        Parameters:
        - edges_to_add: list of tuple (node1, node2), proposed new edges.
        - edges_to_remove: list of tuple (node1, node2), existing edges to be removed.

        Ensures:
        1. Scale-Free networks limit node degree growth to prevent "super hubs".
        2. Small-World networks introduce occasional long-range links.
        3. Nodes maintain a minimum degree to prevent isolation.
        4. Critical bridge edges are preserved.
        5. Neutral nodes do not receive new connections.
        """
        all_nodes = list(self.nodes.keys())

        # Scale-Free Networks need to prevent Super Hubs
        if self.network_type == "scale_free":
            max_degree_limit = np.sqrt(self.N)
            edges_to_add = [(u, v) for u, v in edges_to_add
                            if self.graph.degree[u] < max_degree_limit and self.graph.degree[v] < max_degree_limit]

        # Small-World Networks need to Add Long-Range Links
        if self.network_type == "small_world":
            if nx.is_connected(self.graph):
                avg_shortest_path = nx.average_shortest_path_length(self.graph)
            else:
                avg_shortest_path = 1  # Safe fallback
            
            for node_id in self.nodes:
                if np.random.rand() < 0.1:  # 10% chance to add a long-range link
                    potential_long_range = list(set(all_nodes) - set(self.graph.neighbors(node_id)) - {node_id})
                    if potential_long_range:
                        long_range_node = np.random.choice(potential_long_range)
                        if nx.has_path(self.graph, node_id, long_range_node) and \
                                nx.shortest_path_length(self.graph, node_id, long_range_node) > avg_shortest_path * 1.5:
                            edges_to_add.append((node_id, long_range_node))

        # Ensure Minimum Degree for Connectivity (but not for random)
        if self.network_type in ["small_world", "scale_free"]:
            min_degree = 2
            edges_to_remove = [(u, v) for u, v in edges_to_remove
                            if self.graph.degree[u] > min_degree and self.graph.degree[v] > min_degree]

            # Preserve Critical Bridges (but not for random)
            if edges_to_remove:
                edge_centrality = nx.edge_betweenness_centrality(self.graph, k=min(len(edges_to_remove), 100))  # Optimize calculation
                edges_to_remove = sorted(edges_to_remove, key=lambda e: edge_centrality.get(e, 0), reverse=True)[:int(len(edges_to_remove) * 0.7)]

        # Prevent Neutral Node Connections
        edges_to_add = [(u, v) for u, v in edges_to_add if u not in self.neutral_nodes and v not in self.neutral_nodes]

        # Apply Changes
        self.modify_edges(edges_to_add=edges_to_add, edges_to_remove=edges_to_remove)
        self.save_network_state()


    def adjust_edges_based_on_similarity(self, threshold_connect=0.75, threshold_disconnect=0.25, 
                                     sample_size_connect=5, sample_size_drop=5, 
                                     alpha_user=1.0, alpha_recommendation=1.0,
                                     use_hybrid_recommendation=False,
                                     use_attribute_and_dissimilar_recommendation=False,
                                     use_random_recommendation=False, 
                                     use_content_based_recommendation=False,
                                     use_dissimilar_recommendation=False, 
                                     use_collaborative_recommendation=False,
                                     use_attribute_vector_recommendation=False,
                                     sample_size_hybrid=5,
                                     sample_size_attribute_dissimilar=5, 
                                     sample_size_random=5, 
                                     sample_size_content=5, 
                                     sample_size_dissimilar=5, 
                                     sample_size_collaborative=5,
                                     sample_size_attribute_vector=5):
        """
        Adjusts the network by adding and removing edges based on similarity thresholds and recommendation algorithms.

        Parameters:
        - threshold_connect (float): Minimum similarity score to form a new edge.
        - threshold_disconnect (float): Maximum similarity score to remove an existing edge.
        - sample_size_connect (int): Number of random unconnected nodes each node explores.
        - sample_size_drop (int): Number of existing connections evaluated for removal.
        - alpha_user (float): Scaling factor for similarity calculations.
        - alpha_recommendation (float): Scaling factor for recommendation similarity.
        - use_* (bool): Flags to enable different recommendation strategies.
        - sample_size_* (int): Number of nodes considered for each recommendation strategy.

        Returns:
        - None (modifies the network structure in place).
        """
        edges_to_add = []
        edges_to_remove = []
        all_nodes = list(self.nodes.keys())

        # Iterate over all nodes to evaluate edge modifications
        for node_id in self.nodes:
            if node_id in self.neutral_nodes:
                continue  # Skip neutral nodes

            existing_connections = list(self.graph.neighbors(node_id))
            potential_new_connections = set(all_nodes) - set(existing_connections) - {node_id}

            # Get Recommended Nodes using different algorithms (if using these algorithms)
            recommended_nodes = self.get_recommended_nodes(
                node_id=node_id,
                potential_new_connections=potential_new_connections,
                use_hybrid_recommendation=use_hybrid_recommendation,
                use_attribute_and_dissimilar_recommendation=use_attribute_and_dissimilar_recommendation,
                use_random_recommendation=use_random_recommendation,
                use_content_based_recommendation=use_content_based_recommendation,
                use_dissimilar_recommendation=use_dissimilar_recommendation,
                use_collaborative_recommendation=use_collaborative_recommendation,
                use_attribute_vector_recommendation=use_attribute_vector_recommendation,
                alpha_recommendation=alpha_recommendation,
                sample_size_hybrid=sample_size_hybrid,
                sample_size_attribute_dissimilar=sample_size_attribute_dissimilar,
                sample_size_random=sample_size_random,
                sample_size_content=sample_size_content,
                sample_size_dissimilar=sample_size_dissimilar,
                sample_size_collaborative=sample_size_collaborative,
                sample_size_attribute_vector=sample_size_attribute_vector,
            )

            # Evaluate new connections and determine edges to add
            edges_to_add.extend(self.evaluate_new_connections(
                node_id=node_id, 
                recommended_nodes=recommended_nodes, 
                threshold_connect=threshold_connect, 
                alpha_user=alpha_user
            ))
            # every round all edges (including prev rounds --- keep there, change if still have time left)

            # Restrict edges_to_add to sample_size_connect limit (by default choose 5)
            # if len(edges_to_add) > sample_size_connect:
            #     edges_to_add = [edges_to_add[i] for i in np.random.choice(len(edges_to_add), sample_size_connect, replace=False)]

            # Evaluate existing connections and determine edges to remove
            if sample_size_drop > 0:
                selected_existing_connections = (
                    np.random.choice(existing_connections, sample_size_drop, replace=False) 
                    if len(existing_connections) > sample_size_drop else existing_connections
                )
                edges_to_remove.extend(self.evaluate_existing_connections(
                    node_id=node_id, 
                    existing_connections=selected_existing_connections, 
                    threshold_disconnect=threshold_disconnect, 
                    alpha_user=alpha_user
                ))

        # Apply the network modifications
        self.apply_network_adjustments(edges_to_add, edges_to_remove)


    def modify_edges(self, edges_to_add=None, edges_to_remove=None):
        """
        Dynamically modify the edges in the network.

        Parameters:
        - edges_to_add: list of tuples, each tuple representing an edge (node1, node2) to add.
        - edges_to_remove: list of tuples, each tuple representing an edge (node1, node2) to remove.
        """
        if edges_to_add:
            # Filter out edges that already exist
            edges_to_add = [(u, v) for u, v in edges_to_add if not self.graph.has_edge(u, v)]
            self.graph.add_edges_from(edges_to_add)

        if edges_to_remove:
            # Filter out edges that do not exist
            edges_to_remove = [(u, v) for u, v in edges_to_remove if self.graph.has_edge(u, v)]
            self.graph.remove_edges_from(edges_to_remove)


    def save_network_state(self):
        """
        Save the current state of the network, including node opinions and edges.
        """
        state = {
            'opinions': {node_id: node.get_opinion().tolist() for node_id, node in self.nodes.items()},
            'edges': list(self.graph.edges())
        }
        self.history.append(state)


    def calculate_similarity(self, node_id, other_node_id, alpha=1.0):
        """
        Calculate the similarity between two nodes' opinions.

        Parameters:
        - node_id: int, ID of the first node.
        - other_node_id: int, ID of the second node.

        Returns:
        - float: similarity value between the two nodes' opinions.
        """
        opinion_1 = self.nodes[node_id].get_opinion()
        opinion_2 = self.nodes[other_node_id].get_opinion()
        attr_1 = self.nodes[node_id].get_attributes()
        attr_2 = self.nodes[other_node_id].get_attributes()

        # Compute opinion similarity (cosine similarity)
        if np.linalg.norm(opinion_1) == 0 or np.linalg.norm(opinion_2) == 0:
            opinion_similarity = 0  # Avoid division by zero
        else:
            opinion_similarity = np.dot(opinion_1, opinion_2) / (np.linalg.norm(opinion_1) * np.linalg.norm(opinion_2))
        # fully dependent on opinion
        if alpha == 1.0:
            return opinion_similarity

        # Compute attribute similarity (cosine similarity)
        if len(attr_1) == 0 or len(attr_2) == 0 or np.linalg.norm(attr_1) == 0 or np.linalg.norm(attr_2) == 0:
            attribute_similarity = 0  # If no attribute vector, set similarity to 0
        else:
            attribute_similarity = np.dot(attr_1, attr_2) / (np.linalg.norm(attr_1) * np.linalg.norm(attr_2))

        # Compute final similarity as a weighted sum
        similarity = alpha * opinion_similarity + (1 - alpha) * attribute_similarity

        return similarity
    
    def compute_vector_similarity(self, opinion_vec1, opinion_vec2, attr_vec1, attr_vec2, alpha=1.0):
        """
        Compute similarity between two nodes using their Opinion & Attribute vectors.

        :param opinion_vec1: Opinion vector of node 1.
        :param opinion_vec2: Opinion vector of node 2.
        :param attr_vec1: Attribute vector of node 1.
        :param attr_vec2: Attribute vector of node 2.
        :param alpha: Weight factor for opinion vs. attribute similarity (default = 1.0).
        :return: Similarity score (float).
        """
        opinion_similarity = 0
        if np.linalg.norm(opinion_vec1) > 0 and np.linalg.norm(opinion_vec2) > 0:
            opinion_similarity = np.dot(opinion_vec1, opinion_vec2) / (np.linalg.norm(opinion_vec1) * np.linalg.norm(opinion_vec2))

        attribute_similarity = 0
        if len(attr_vec1) > 0 and len(attr_vec2) > 0 and np.linalg.norm(attr_vec1) > 0 and np.linalg.norm(attr_vec2) > 0:
            attribute_similarity = np.dot(attr_vec1, attr_vec2) / (np.linalg.norm(attr_vec1) * np.linalg.norm(attr_vec2))

        similarity = alpha * opinion_similarity + (1 - alpha) * attribute_similarity
        return similarity
    

    def calculate_all_neighbors_similarity(self, node_id, alpha=1.0):
        """
            Calculates all neighbors' similarity to a given node.

            :param node_id: int, ID of the node.
            :param alpha: Weight factor for opinion vs. attribute similarity (default = 1.0, fully on opinion similarity).
        """
        neighbors = list(self.graph.neighbors(node_id))
        similarity_dict = {}
        for neighbor in neighbors:
            similarity_dict[neighbor] = self.calculate_similarity(node_id, neighbor, alpha)
        return similarity_dict
    

    """ Measurement functions """
    ''' === Echo Chamber Measurement === '''
    ''' === Echo Chamber Measurement === '''
    def compute_avg_neighbor_opinion(self, node):
        """
        Computes the average opinion of a node's neighbors.

        :param node: The node whose neighbors' opinions are averaged.
        :return: The mean opinion vector of the node's neighbors.
        """
        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return self.nodes[node].get_opinion()  # Return the full opinion vector if no neighbors

        neighbor_opinions = np.array([self.nodes[neighbor].get_opinion() for neighbor in neighbors])
        return np.mean(neighbor_opinions, axis=0)  # Compute mean per dimension

    def compute_nearest_community_opinion(self, node):
        """
        Determines the opinion of the nearest community for an isolated node.

        :param node: The isolated node.
        :return: The average opinion vector of the nearest community.
        """
        # Identify connected nodes (non-isolated nodes)
        connected_nodes = [n for n in self.graph if self.graph[n]]
        if not connected_nodes:
            return self.nodes[node].get_opinion()  # Return self opinion if no connected nodes

        community_opinions = np.array([self.nodes[n].get_opinion() for n in connected_nodes])
        return np.mean(community_opinions, axis=0)  # Compute mean per dimension

    def compute_echo_chamber_extent(self):
        """
        Computes the improved E'_chamber metric for measuring polarization in the social network.

        Formula:
            E'_chamber = (N_c/N) * (1/N_c) * sum( abs(mean_neighbor - opinion).sum()/d )
                      + (N_iso/N) * (1/N_iso) * sum( abs(mean_community - opinion).sum()/d )

        :return: The computed E'_chamber value.
        """
        connected_nodes = [node for node in self.graph if len(self.graph[node]) > 0]
        isolated_nodes = [node for node in self.graph if not self.graph[node]]

        N_c = len(connected_nodes)
        N_iso = len(isolated_nodes)
        N = N_c + N_iso  # Total number of nodes

        if N == 0:
            return 0.0  # Avoid division by zero in an empty graph

        # Compute E_conn: Neighbor alignment for connected nodes
        E_conn = 0
        if N_c > 0:
            E_conn = sum(
                np.abs(self.compute_avg_neighbor_opinion(node) - self.nodes[node].get_opinion()).sum() / len(self.nodes[node].get_opinion())
                for node in connected_nodes
            ) / N_c

        # Compute E_iso: Community alignment for isolated nodes
        E_iso = 0
        if N_iso > 0:
            E_iso = sum(
                np.abs(self.nodes[node].get_opinion() - self.compute_nearest_community_opinion(node)).sum() / len(self.nodes[node].get_opinion())
                for node in isolated_nodes
            ) / N_iso

        # Compute final metric
        E_chamber_prime = (N_c * E_conn + N_iso * E_iso) / N
        result = 1 - E_chamber_prime

        return result
    
    def compute_average_degree(self):
        """
        Computes the average degree of the network.

        The degree of a node is the number of connections it has.
        The average degree is computed as the sum of all node degrees divided by the total number of nodes.

        :return: The average degree of the network.
        """
        if len(self.graph) == 0:
            return 0.0  # Avoid division by zero for an empty network

        total_degree = sum(dict(self.graph.degree()).values())
        return total_degree / len(self.graph)


    ''' === Modularity === '''
    def assign_isolated_nodes(self, community_labels, alpha=0.5):
        """
        Assigns isolated nodes to the closest community based on similarity.

        :param community_labels: Dictionary of node -> community mapping.
        :param alpha: Weight factor for opinion vs. attribute similarity (default=0.5).
        """
        isolated_nodes = [node for node in self.graph if not self.graph[node]]
        connected_nodes = [node for node in self.graph if self.graph[node]]

        # If no connected nodes, leave isolated nodes as their own community
        if not connected_nodes:
            return

        # Compute community centers (Opinion Vector & Attribute Vector)
        community_centers = {}
        community_attributes = {}

        for node in connected_nodes:
            comm = community_labels[node]
            if comm not in community_centers:
                community_centers[comm] = []
                community_attributes[comm] = []

            community_centers[comm].append(self.nodes[node].get_opinion())
            community_attributes[comm].append(self.nodes[node].get_attributes())

        # Compute mean vectors for each community
        community_avg = {comm: np.mean(vectors, axis=0) for comm, vectors in community_centers.items()}
        community_attr_avg = {comm: np.mean(attrs, axis=0) if len(attrs) > 0 else np.zeros_like(attrs[0])
                            for comm, attrs in community_attributes.items()}

        # Assign isolated nodes to nearest community
        for node in isolated_nodes:
            node_opinion = self.nodes[node].get_opinion()
            node_attr = self.nodes[node].get_attributes()

            best_community = None
            best_similarity = -1

            for comm, avg_opinion in community_avg.items():
                avg_attr = community_attr_avg[comm]  # Use mean attribute vector
                similarity = self.compute_vector_similarity(node_opinion, avg_opinion, node_attr, avg_attr, alpha)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_community = comm

            if best_community is not None:
                community_labels[node] = best_community
            else:
                community_labels[node] = max(community_labels.values()) + 1  # Create new community if no match


    def compute_modularity(self, alpha=1.0):
        """
        Computes the modularity score of the current network, handling isolated nodes.

        :param alpha: Weight factor for opinion vs. attribute similarity (default=0.5).
        :return: Modularity score.
        """
        # Step 1: Detect communities using Louvain method (greedy modularity)
        communities = list(greedy_modularity_communities(self.graph))

        # Step 2: Convert to node-community mapping
        community_labels = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_labels[node] = i

        # Step 3: Handle isolated nodes
        self.assign_isolated_nodes(community_labels, alpha)

        # Step 4: Convert mapping back to list format
        final_communities = {}
        for node, comm in community_labels.items():
            if comm not in final_communities:
                final_communities[comm] = set()
            final_communities[comm].add(node)

        final_communities = list(final_communities.values())

        # Step 5: Compute modularity
        return modularity(self.graph, final_communities)


    def measure_polarization(self):
        """
        Calculate the polarization level as the standard deviation of opinions.
        """

        """
        Measures the polarization level of the group based on the standard deviation of opinions.

        Formula:
            P = SD(x_0, x_1, ..., x_N)
            - self.nodes.values() for opinions (x_0, x_1, ..., x_N)

        Returns:
        - polarization_level: float, standard deviation of opinions, representing polarization.

        Interpretation:
        - P: Polarization level, measured by the standard deviation (SD) of individual opinions.
        - The greater the P, the less polarized the group is (opinions are more similar).
        """
        opinions = np.array([node.get_opinion() for node in self.nodes.values()])
        # Standard deviation for each dimension
        polarization_per_dim = np.std(opinions, axis=0)
        return np.mean(polarization_per_dim)


    def measure_radicalization(self):
        """
        Measures the radicalization level of the group based on the average absolute value of opinions.

        Formula:
            R = (1/N) * Σ|x_i| for i = 1 to N
            - self.nodes.values() for opinions (x_0, x_1, ..., x_N)

        Returns:
        - radicalization_level: float, average of absolute opinion values, representing radicalization.

        Interpretation:
        - R: Radicalization level, calculated as the mean absolute value of opinions.
        - The greater the R, the more extreme the opinions of individuals in the group are.
        """
        opinions = np.array([node.get_opinion() for node in self.nodes.values()])
        # Mean absolute value for each dimension
        radicalization_per_dim = np.mean(np.abs(opinions), axis=0)
        return np.mean(radicalization_per_dim)


    def measure_density(self):
        """
        Measure the density of the network.

        Formula:
            D = 2E / (N * (N - 1))

        Parameters:
        - E: int, number of edges in the network (self.graph.number_of_edges()).
        - N: int, number of nodes in the network (self.graph.number_of_nodes()).

        Returns:
        - float: Network density, representing the proportion of actual edges
                relative to all possible edges in the network.
        """
        E = self.graph.number_of_edges()  # Total number of edges in the network
        N = self.graph.number_of_nodes()  # Total number of nodes in the network
        return 2 * E / (N * (N - 1)) if N > 1 else 0  # Return density or 0 if only one node


    def measure_clustering_coefficient(self):
        """
        Measure the global clustering coefficient of the network.

        Formula:
            C = Number of Closed Triplets / Number of Connected Triplets

        Explanation of nx.transitivity:
        - NetworkX's `nx.transitivity` calculates the global clustering coefficient of the graph.
        - A "triplet" consists of three nodes that are connected by either two or three edges.
        - A "closed triplet" forms a triangle (all three nodes are connected).
        - The clustering coefficient measures the tendency of nodes to form tightly-knit groups.

        Parameters:
        - self.graph: NetworkX Graph object representing the social network.

        Returns:
        - float: Global clustering coefficient of the network, a value between 0 and 1.
                - 0: No clustering (nodes are not part of closed triplets).
                - 1: Fully clustered network (every triplet forms a triangle).
        """
        return nx.transitivity(self.graph)  # Calculate and return the clustering coefficient


    def compute_search_frequency(self, node_id):
        """
        Compute the search frequency for a given node.
        - Low-degree nodes search more frequently.
        - High-degree nodes (Hub nodes) search less frequently.

        Returns:
        - search_frequency: int, number of rounds before this node searches for new connections again.
        """
        node_degree = self.graph.degree[node_id]
        search_frequency = int(20 / (node_degree + 1))  # High-degree nodes have lower search frequency
        search_frequency = max(3, min(20, search_frequency))
        return search_frequency

    def compute_disconnect_frequency(self, node_id):
        """
        Compute the disconnect frequency for a given node.
        - Nodes with lower interaction scores are more likely to lose connections.
        - High-interaction nodes retain connections longer.

        Returns:
        - disconnect_frequency: int, number of rounds before this node evaluates disconnections again.
        """
        interaction_score = np.random.rand()  # Simulate interaction level (0 to 1)
        disconnect_frequency = int(50 / (interaction_score * 10 + 1))  # Higher interaction → lower disconnect rate
        disconnect_frequency = max(3, min(20, disconnect_frequency))
        return max(3, min(20, disconnect_frequency))


    def content_based_recommendation(self, node_id, top_n=5):
        """
        Generate recommendations for a specific node based on similarity of opinions.

        Parameters:
        - node_id: int, the node for which recommendations are being generated.
        - top_n: int, number of recommendations per node.

        Returns:
        - list: `top_n` recommended node IDs for the given node.
        """
        # Compute similarity matrix
        opinion_matrix = np.array([node.get_opinion() for node in self.nodes.values()])
        norms = np.linalg.norm(opinion_matrix, axis=1, keepdims=True)
        normalized_matrix = opinion_matrix / norms
        similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

        # Get already connected nodes
        existing_connections = set(self.graph.neighbors(node_id))

        # Compute similarity scores
        similarity_scores = similarity_matrix[node_id]
        similar_nodes = [
            (i, score) for i, score in enumerate(similarity_scores)
            if i != node_id and i not in existing_connections  # Exclude already connected nodes
        ]

        # Sort nodes by similarity in descending order and return top-N
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in similar_nodes[:top_n]]

    def random_recommendation(self, node_id, potential_new_connections, top_n=5):
        """
        Generate random recommendations for a specific node.

        Parameters:
        - node_id: int, the node for which recommendations are being generated.
        - top_n: int, number of recommendations per node.
        - seed: int or None, seed for the random number generator.

        Returns:
        - list: `top_n` randomly recommended node IDs for the given node.
        """
        possible_nodes = list(potential_new_connections)

        if len(possible_nodes) >= top_n:
            return list(np.random.choice(possible_nodes, top_n, replace=False))
        else:
            return possible_nodes

    def dissimilar_recommendation(self, node_id, top_n=5):
        """
        Generate recommendations for a specific node based on opinion dissimilarity.

        Parameters:
        - node_id: int, the node for which recommendations are being generated.
        - top_n: int, number of recommendations per node.

        Returns:
        - list: `top_n` recommended node IDs for the given node.
        """
        existing_connections = set(self.graph.neighbors(node_id))

        # Compute dissimilarity scores (1 - similarity)
        dissimilar_nodes = [
            (other_node_id, 1 - self.calculate_similarity(node_id, other_node_id))
            for other_node_id in self.nodes
            if node_id != other_node_id and other_node_id not in existing_connections  # Exclude connected nodes
        ]

        # Sort nodes by dissimilarity and return top-N
        dissimilar_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in dissimilar_nodes[:top_n]]

    def collaborative_filtering_recommendation(self, node_id, top_n=5):
        """
        Generate recommendations for a specific node using Collaborative Filtering (common neighbors).

        Parameters:
        - node_id: int, the node for which recommendations are being generated.
        - top_n: int, number of recommendations per node.

        Returns:
        - list: `top_n` recommended node IDs for the given node.
        """
        neighbors = set(self.graph.neighbors(node_id))
        similarities = []

        for other_node_id in self.nodes:
            if node_id == other_node_id or other_node_id in neighbors:
                continue  # Skip self and already connected nodes

            # Get neighbors of the other node
            other_neighbors = set(self.graph.neighbors(other_node_id))

            # Compute Jaccard similarity: |Intersection| / |Union|
            intersection = len(neighbors & other_neighbors)
            union = len(neighbors | other_neighbors)
            similarity = intersection / union if union > 0 else 0

            # Maintain top-N highest similarities using heapq
            if len(similarities) < top_n:
                heapq.heappush(similarities, (similarity, other_node_id))
            else:
                heapq.heappushpop(similarities, (similarity, other_node_id))

        # Extract sorted top-N recommendations
        return [node for _, node in sorted(similarities, reverse=True)]
    
    def recommend_by_attribute_vector(self, node_id, top_n=5):
        """
        Generate recommendations for a specific node based on similarity of attribute vectors.

        Parameters:
        - node_id: int, the node for which recommendations are being generated.
        - top_n: int, number of recommendations per node.

        Returns:
        - list: `top_n` recommended node IDs for the given node.
        """
        # Retrieve all nodes' attribute vectors and convert them into a NumPy array
        attribute_matrix = np.array([
            node.get_attributes() if node.get_attributes() is not None else np.zeros_like(self.nodes[node_id].get_attributes())
            for node in self.nodes.values()
        ])

        # Normalize the attribute vectors
        norms = np.linalg.norm(attribute_matrix, axis=1, keepdims=True)
        normalized_matrix = np.where(norms == 0, 0, attribute_matrix / norms)  # Prevent division by zero
        similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)  # Compute cosine similarity

        # Retrieve already connected nodes to avoid recommending them again
        existing_connections = set(self.graph.neighbors(node_id))

        # Compute similarity scores
        similarity_scores = similarity_matrix[node_id]
        similar_nodes = [
            (i, score) for i, score in enumerate(similarity_scores)
            if i != node_id and i not in existing_connections  # Exclude already connected nodes
        ]

        # Sort nodes by similarity in descending order and return the top-N
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in similar_nodes[:top_n]]


    def attribute_and_dissimilar_recommendation(self, node_id, top_n=5, alpha_recommendation=1.0):
        """
        Generate recommendations based on opinion dissimilarity and attribute similarity.

        Parameters:
        - node_id: int, the node for which recommendations are being generated.
        - top_n: int, number of recommendations per node.
        - alpha_recommendation: float, weight for opinion dissimilarity (default: 1.0).
          - alpha_recommendation = 1.0 → considers only opinion dissimilarity
          - alpha_recommendation = 0.0 → considers only attribute similarity
          - 0.0 < alpha_recommendation < 1.0 → a combination of both

        Returns:
        - list: `top_n` recommended node IDs for the given node.
        """
        existing_connections = set(self.graph.neighbors(node_id))  # Get existing connections
        all_node_ids = list(self.nodes.keys())

        # === Compute opinion similarity using cosine similarity (matrix multiplication) ===
        opinion_matrix = np.array([node.get_opinion() for node in self.nodes.values()])
        opinion_norms = np.linalg.norm(opinion_matrix, axis=1, keepdims=True)
        normalized_opinion_matrix = opinion_matrix / (opinion_norms + 1e-10)  # Avoid division by zero
        opinion_sim_matrix = np.dot(normalized_opinion_matrix, normalized_opinion_matrix.T)

        opinion_dissimilarity = 1 - opinion_sim_matrix  # Convert similarity to dissimilarity

        # === Compute attribute similarity using cosine similarity (matrix multiplication) ===
        attribute_matrix = np.array([node.get_attributes() for node in self.nodes.values()])
        attribute_norms = np.linalg.norm(attribute_matrix, axis=1, keepdims=True)

        # Handle zero-vector attributes by setting norms to 1 (prevents NaN)
        attribute_norms[attribute_norms == 0] = 1

        normalized_attribute_matrix = attribute_matrix / attribute_norms
        attribute_sim_matrix = np.dot(normalized_attribute_matrix, normalized_attribute_matrix.T)

        # === Compute final recommendation scores ===
        final_scores = alpha_recommendation * opinion_dissimilarity[node_id] + (1 - alpha_recommendation) * attribute_sim_matrix[node_id]

        # Exclude self and already connected nodes
        recommendations = [
            (other_node_id, final_scores[other_node_id])
            for other_node_id in all_node_ids
            if other_node_id != node_id and other_node_id not in existing_connections
        ]

        # Sort by score in descending order and select top-N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in recommendations[:top_n]]
    

    def recommend_hybrid(self, node_id, top_n=5, alpha_recommendation=1.0):
        """
        Generate recommendations based on a weighted combination of opinion similarity and attribute similarity.

        Parameters:
        - node_id: int, the node for which recommendations are being generated.
        - top_n: int, number of recommendations per node.
        - alpha_recommendation: float, weight for opinion similarity (default: 0.5).
            - alpha_recommendation = 1.0 → considers only opinion similarity
            - alpha_recommendation = 0.0 → considers only attribute similarity
            - 0.0 < alpha_recommendation < 1.0 → a combination of both

        Returns:
        - list: `top_n` recommended node IDs for the given node.
        """

        existing_connections = set(self.graph.neighbors(node_id))  # Get existing connections
        all_node_ids = list(self.nodes.keys())

        # === Compute opinion similarity using cosine similarity ===
        opinion_matrix = np.array([node.get_opinion() for node in self.nodes.values()])
        opinion_norms = np.linalg.norm(opinion_matrix, axis=1, keepdims=True)
        normalized_opinion_matrix = opinion_matrix / (opinion_norms + 1e-10)  # Avoid division by zero
        opinion_sim_matrix = np.dot(normalized_opinion_matrix, normalized_opinion_matrix.T)

        # === Compute attribute similarity using cosine similarity ===
        attribute_matrix = np.array([
            node.get_attributes() if node.get_attributes() is not None else np.zeros_like(self.nodes[node_id].get_attributes())
            for node in self.nodes.values()
        ])
        attribute_norms = np.linalg.norm(attribute_matrix, axis=1, keepdims=True)

        # Prevent division by zero for zero vectors
        attribute_norms[attribute_norms == 0] = 1

        normalized_attribute_matrix = attribute_matrix / attribute_norms
        attribute_sim_matrix = np.dot(normalized_attribute_matrix, normalized_attribute_matrix.T)

        # === Compute final recommendation scores using weighted similarity ===
        final_scores = alpha_recommendation * opinion_sim_matrix[node_id] + (1 - alpha_recommendation) * attribute_sim_matrix[node_id]

        # Exclude self and already connected nodes
        recommendations = [
            (other_node_id, final_scores[other_node_id])
            for other_node_id in all_node_ids
            if other_node_id != node_id and other_node_id not in existing_connections
        ]

        # Sort by score in descending order and select top-N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in recommendations[:top_n]]


    def draw_network(self, title="Network with Social Circles"):
        """
        Visualize the network with nodes grouped and colored by their identified social circles.

        Parameters:
        - title: str, title for the plot.
        """
        # Detect communities (social circles)
        communities = list(greedy_modularity_communities(self.graph))  # Find communities
        community_map = {}
        for idx, community in enumerate(communities):
            for node in community:
                community_map[node] = idx

        # Create a layout grouping nodes by community
        pos = {}
        community_positions = {}  # Center positions for each community
        center_x, center_y = 0, 0
        radius_increment = 2.0  # Distance between communities
        for idx, community in enumerate(communities):
            # Calculate a circular layout for each community
            angle_step = 2 * np.pi / len(community)
            community_center = (center_x, center_y)
            community_positions[idx] = community_center

            for i, node in enumerate(community):
                angle = i * angle_step
                pos[node] = (
                    community_center[0] + np.cos(angle),
                    community_center[1] + np.sin(angle),
                )

            # Final Center for the next community
            center_x += radius_increment

        # Assign colors to nodes based on each's community, and Visualize
        node_colors = [community_map[node_id] for node_id in self.graph.nodes()]
        fig, ax = plt.subplots(figsize=(12, 9))
        nx.draw(
            self.graph, pos=pos,
            node_color=node_colors,
            cmap=plt.cm.Set3,
            with_labels=True,
            node_size=500,
            ax=ax
        )
        plt.title(title)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Set3)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Community Index")
        plt.show()


    def draw_network2(self, title="Network with Social Circles"):
        """
        Visualize the network with nodes grouped and colored by their identified social circles.

        Parameters:
        - title: str, title for the plot.
        """
        # **1️⃣ Detect communities**
        communities = list(greedy_modularity_communities(self.graph))  # Detect communities
        community_map = {}  # Store community index for each node
        for idx, community in enumerate(communities):
            for node in community:
                community_map[node] = idx

        # **2️⃣ Choose an appropriate layout**
        if self.network_type == "random":
            pos = nx.spring_layout(self.graph, seed=self.seed)
        elif self.network_type == "scale_free":
            pos = nx.kamada_kawai_layout(self.graph)  # Suitable for scale-free networks, emphasizing hubs
        elif self.network_type == "small_world":
            pos = nx.circular_layout(self.graph)  # Highlights community structure in small-world networks

        # **3️⃣ Compute node sizes**
        node_sizes = [300] * self.N  # Default size
        if self.network_type == "scale_free":
            max_degree = max(dict(self.graph.degree).values())  # Highest node degree
            node_sizes = [100 + 900 * (self.graph.degree[node] / max_degree) for node in self.graph.nodes()]

        # **4️⃣ Assign colors**
        community_colors = [community_map[node_id] for node_id in self.graph.nodes()]

        # **5️⃣ Identify long-range links for small-world networks**
        # long_range_edges = []
        # if self.network_type == "small_world":
        #     avg_shortest_path = nx.average_shortest_path_length(self.graph)  # Compute average shortest path
        #     for u, v in self.graph.edges():
        #         if nx.shortest_path_length(self.graph, u, v) > avg_shortest_path * 1.5:
        #             long_range_edges.append((u, v))  # Mark as a long-range link
        # Identify long-range links for small-world networks
        long_range_edges = []
        if self.network_type == "small_world":
            if nx.is_connected(self.graph):  # Only compute if the graph is connected
                avg_shortest_path = nx.average_shortest_path_length(self.graph)
                for u, v in self.graph.edges():
                    if nx.shortest_path_length(self.graph, u, v) > avg_shortest_path * 1.5:
                        long_range_edges.append((u, v))  # Mark as a long-range link
        else:
            print(f"⚠️ Warning: Small-World Network is not connected, skipping long-range link analysis.")


        # **6️⃣ Draw the network**
        fig, ax = plt.subplots(figsize=(12, 9))

        # **(a) Draw regular edges**
        nx.draw_networkx_edges(self.graph, pos, ax=ax, alpha=0.5, edge_color="gray")

        # **(b) Highlight long-range connections in small-world networks**
        if self.network_type == "small_world":
            nx.draw_networkx_edges(self.graph, pos, edgelist=long_range_edges, edge_color="red", width=2, ax=ax)

        # **(c) Draw nodes**
        nx.draw(
            self.graph, pos=pos,
            node_color=community_colors, cmap=plt.cm.Set3,  # Assign colors based on communities
            with_labels=True, node_size=node_sizes, ax=ax
        )

        # **7️⃣ Add color legend**
        plt.title(title)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Set3)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Community Index")
        plt.show()


    def visualize_neutral_nodes(self):
        """
        Visualizes the social network with neutral nodes highlighted in red and regular nodes in blue.
        """
        print("🔹 Visualizing Neutral Nodes in the Network...")

        # Generate layout for visualization
        pos = nx.spring_layout(self.graph, seed=self.seed)

        # Assign colors: Neutral nodes = Red, Regular nodes = Blue
        node_colors = ["red" if node in self.neutral_nodes else "blue" for node in self.graph.nodes()]

        # Draw the network
        plt.figure(figsize=(10, 7))
        nx.draw(self.graph, pos, node_color=node_colors, with_labels=True, node_size=100)
        plt.title("Social Network with Neutral Nodes (Red)")
        plt.show()


    def plot_metrics_over_time(self, metrics_to_plot=["density", "polarization", "radicalization"]):
        """
        Plot selected metrics over time based on recorded network states.

        Parameters:
        - metrics_to_plot: list of str, metrics to include in the plot. Options: "density", "polarization", "radicalization".
        """
        # Prepare data for plotting
        densities = [
            2 * len(state['edges']) / (self.N * (self.N - 1)) if self.N > 1 else 0
            for state in self.history
        ] if "density" in metrics_to_plot else None
        polarizations = [np.std([op for op_list in state['opinions'].values() for op in op_list]) for state in self.history] if "polarization" in metrics_to_plot else None
        radicalizations = [np.mean([np.abs(op) for op_list in state['opinions'].values() for op in op_list]) for state in self.history] if "radicalization" in metrics_to_plot else None

        # Plot metrics
        plt.figure(figsize=(12, 6))
        if densities:
            plt.plot(densities, label="Density")
        if polarizations:
            plt.plot(polarizations, label="Polarization")
        if radicalizations:
            plt.plot(radicalizations, label="Radicalization")

        plt.xlabel("Time Step")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.title("Metrics Over Time")
        plt.show()


    def save_history_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.history, f)


    def load_history_from_file(self, filename):
        with open(filename, 'r') as f:
            self.history = json.load(f)


    ''' check how cross-community connection can solve polarization '''
    def strategy_based_disconnect(self, num_communities=2, num_edges=3, intra_community=True):
        """
        Strategically removes edges from selected communities.

        Parameters:
        - num_communities (int): Number of communities to perform disconnection.
        - num_edges (int): Number of edges to remove per selected community.
        - intra_community (bool):
            - If True, removes edges **within** selected communities.
            - If False, removes edges **between** different communities (cross-community).

        Effect:
        - Updates the graph by removing edges, ensuring no community is completely disconnected.
        """

        # Step 1: Identify communities
        communities = list(greedy_modularity_communities(self.graph))
        num_communities = min(num_communities, len(communities))  # Ensure we don't select more than available

        # Step 2: Randomly choose communities to perform edge removal
        selected_communities = np.random.choice(len(communities), num_communities, replace=False)

        for community_index in selected_communities:
            community_nodes = list(communities[community_index])
            edges_to_remove = []

            if intra_community:
                # Step 3a: Select edges **within** the community
                intra_edges = [(u, v) for u, v in self.graph.edges() if u in community_nodes and v in community_nodes]
            else:
                # Step 3b: Select edges **between** this community and others
                intra_edges = [(u, v) for u, v in self.graph.edges() if (u in community_nodes) ^ (v in community_nodes)]

            if len(intra_edges) > 0:
                chosen_indices = np.random.choice(range(len(intra_edges)), min(num_edges, len(intra_edges)), replace=False)
                edges_to_remove = [intra_edges[i] for i in chosen_indices]
            else:
                print("⚠️ No intra-community edges available to disconnect!")
                edges_to_remove = []


            # Step 4: Remove selected edges
            for edge in edges_to_remove:
                self.graph.remove_edge(*edge)
                print(f"❌ Removed Edge {edge} (Intra-Community: {intra_community})")


    def strategy_based_connect(self, num_communities=2, num_edges=3, intra_community=True):
        """
        Strategically adds edges within or between selected communities.

        Parameters:
        - num_communities (int): Number of communities to perform connection.
        - num_edges (int): Number of edges to add per selected community.
        - intra_community (bool):
            - If True, connects nodes **within** selected communities.
            - If False, connects **across different communities**.

        Effect:
        - Updates the graph by adding edges strategically.
        """

        # Step 1: Identify communities
        communities = list(greedy_modularity_communities(self.graph))
        num_communities = min(num_communities, len(communities))  # Ensure we don't exceed available communities

        # Step 2: Randomly choose communities to perform edge addition
        selected_communities = np.random.choice(len(communities), num_communities, replace=False)

        for community_index in selected_communities:
            community_nodes = list(communities[community_index])
            edges_to_add = []

            if intra_community:
                # Step 3a: Select nodes within the same community for new connections
                potential_edges = [(u, v) for u in community_nodes for v in community_nodes if u != v and not self.graph.has_edge(u, v) and not self.graph.has_edge(v, u)]
            else:
                # Step 3b: Select cross-community nodes for new connections
                other_communities = [i for i in range(len(communities)) if i != community_index]
                other_community_index = np.random.choice(other_communities)
                other_community_nodes = list(communities[other_community_index])

                # Connect highest-degree nodes across communities
                top_nodes_a = sorted(community_nodes, key=lambda node: self.graph.degree[node], reverse=True)[:num_edges]
                top_nodes_b = sorted(other_community_nodes, key=lambda node: self.graph.degree[node], reverse=True)[:num_edges]

                potential_edges = [(u, v) for u in top_nodes_a for v in top_nodes_b if not self.graph.has_edge(u, v) and not self.graph.has_edge(v, u)]

            # Ensure we do not attempt to add more edges than available
            if len(potential_edges) > 0:
                #edges_to_add = np.random.choice(potential_edges, min(num_edges, len(potential_edges)), replace=False)
                edges_to_add = np.random.choice(range(len(potential_edges)), min(num_edges, len(potential_edges)), replace=False)
                edges_to_add = [potential_edges[i] for i in edges_to_add]
            else:
                print(f"⚠️ No valid edges found for connection (num_communities={num_communities}, num_edges={num_edges})")
                edges_to_add = []

            # Step 4: Add selected edges
            for edge in edges_to_add:
                if not self.graph.has_edge(*edge):
                    self.graph.add_edge(*edge)
                    print(f"✅ Added Edge {edge} (Intra-Community: {intra_community})")
                else:
                    print(f"⚠️ Edge {edge} already exists, skipping.")


    """ Testing function """
    def count_cross_community_edges(self, communities):
        """
        Counts the number of edges that connect nodes between different communities.

        Parameters:
        - communities (list of sets): A list where each set contains the nodes of a community.

        Returns:
        - int: The number of cross-community edges.
        """
        cross_edges = 0
        community_map = {}  # Map each node to its community index

        for idx, community in enumerate(communities):
            for node in community:
                community_map[node] = idx

        for u, v in self.graph.edges():
            if community_map.get(u) != community_map.get(v):  # Check if nodes belong to different communities
                cross_edges += 1

        return cross_edges
    

    ''' Histogram Related Functions '''
    def reduce_opinion_dimension(self, opinion, n_components=2):
        """
        Perform PCA on a single opinion vector.

        Parameters:
        - opinion (np.ndarray): 1D array of length d (opinion vector).
        - n_components (int): Number of dimensions to reduce to (default: 2).

        Returns:
        - np.ndarray: Transformed opinion with shape (n_components,).
        """
        opinion = np.array(opinion).reshape(1, -1)  # Ensure it's 2D for PCA
        if opinion.shape[1] > n_components:
            pca = PCA(n_components=n_components)
            return pca.fit_transform(opinion)[0]  # Return as 1D array
        return opinion.flatten()  # If already 2D or lower, return as-is

    def get_reduced_opinions(self, n_components=2):
        """
        Apply PCA to all opinion vectors in the network.

        Returns:
        - np.ndarray: Reduced opinions with shape (N, n_components)
        """
        opinions = np.array([self.nodes[n].get_opinion() for n in self.nodes])  # (N, d)
        if opinions.shape[1] > n_components:
            pca = PCA(n_components=n_components)
            return pca.fit_transform(opinions)  # (N, n_components)
        return opinions  # If already low-dimension, return as is

