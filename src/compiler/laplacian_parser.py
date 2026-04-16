import json
import numpy as np

class LaplacianParser:
    def __init__(self, trained_model_path: str):
        self.trained_model_path = trained_model_path
        self._load_model()
        self._extract_topology()

    def _load_model(self):
        with open(self.trained_model_path, 'r') as f:
            self.model_data = json.load(f)
            
        self.trees = self.model_data['learner']['gradient_booster']['model']['trees']
        self.n_trees = len(self.trees)
        self.n_features = self.model_data['learner']['learner_model_param']['num_feature']
        
        # Find the maximum number of nodes across all trees to pad our matrices uniformly
        self.max_nodes = max(len(tree['left_children']) for tree in self.trees)
        
    def _extract_topology(self):
        """
        Extracts the raw tree arrays and pads them to max_nodes.
        """
        self.left_children = np.full((self.n_trees, self.max_nodes), -1, dtype=np.int32)
        self.right_children = np.full((self.n_trees, self.max_nodes), -1, dtype=np.int32)
        self.split_indices = np.zeros((self.n_trees, self.max_nodes), dtype=np.int32)
        self.split_conditions = np.zeros((self.n_trees, self.max_nodes), dtype=np.float32)
        
        for i, tree in enumerate(self.trees):
            n_nodes = len(tree['left_children'])
            self.left_children[i, :n_nodes] = tree['left_children']
            self.right_children[i, :n_nodes] = tree['right_children']
            self.split_indices[i, :n_nodes] = tree['split_indices']
            self.split_conditions[i, :n_nodes] = tree['split_conditions']

    def get_routing_matrices(self):
        """
        Builds the Phase 2 components: W (Routing) and tau (Thresholds).
        We still need these to convert the input data X into probabilities.
        """
        W = np.zeros((self.n_trees, self.max_nodes, int(self.n_features)), dtype=np.float32)
        for i in range(self.n_trees):
            for j in range(self.max_nodes):
                if self.left_children[i, j] != -1:  # Internal node
                    feat_idx = self.split_indices[i, j]
                    W[i, j, feat_idx] = 1.0
                    
        W_transposed = W.transpose(0, 2, 1)  # Shape: [trees, features, nodes]
        return W_transposed, self.split_conditions

    def get_graph_topology(self):
        """
        Builds the Adjacency structures for Phase 3 (Laplacian Inference).
        Returns the data needed to build both Dense and CSR matrices.
        """
        # 1. The Dense Adjacency Template: Shape [trees, nodes, nodes]
        # We use 1 to mark a Left edge, and 2 to mark a Right edge.
        dense_A_template = np.zeros((self.n_trees, self.max_nodes, self.max_nodes), dtype=np.int32)
        
        # 2. The CSR Topology arrays (Batched across trees)
        # Max out-edges per tree is 2 * (max_nodes - 1)
        max_edges = self.max_nodes * 2 
        csr_indices = np.zeros((self.n_trees, max_edges), dtype=np.int32)
        csr_indptr = np.zeros((self.n_trees, self.max_nodes + 1), dtype=np.int32)
        
        # 3. Leaf Weights (To extract the final signal)
        leaf_weights = np.zeros((self.n_trees, self.max_nodes), dtype=np.float32)
        
        for tree_idx in range(self.n_trees):
            edge_counter = 0
            
            for node_idx in range(self.max_nodes):
                # Set the row pointer for the current node
                csr_indptr[tree_idx, node_idx] = edge_counter
                
                left = self.left_children[tree_idx, node_idx]
                right = self.right_children[tree_idx, node_idx]
                
                if left != -1 and right != -1:
                    # Internal Node: Mark the edges
                    dense_A_template[tree_idx, node_idx, left] = 1
                    dense_A_template[tree_idx, node_idx, right] = 2
                    
                    # Add to CSR arrays
                    csr_indices[tree_idx, edge_counter] = left
                    edge_counter += 1
                    csr_indices[tree_idx, edge_counter] = right
                    edge_counter += 1
                elif left == -1:
                    # Leaf Node: Capture the weight (split_conditions stores leaf weights for leaves)
                    leaf_weights[tree_idx, node_idx] = self.split_conditions[tree_idx, node_idx]
            
            # Cap the final row pointer
            csr_indptr[tree_idx, self.max_nodes] = edge_counter
            
        return {
            'dense_A_template': dense_A_template,
            'csr_indices': csr_indices,
            'csr_indptr': csr_indptr,
            'leaf_weights': leaf_weights,
            'is_leaf_mask': (self.left_children == -1)
        }
    

# lap_parser = LaplacianParser(trained_model_path="checkpoints/xgboost_model/xgboost_model.json")
# routing_matrices = lap_parser.get_routing_matrices()