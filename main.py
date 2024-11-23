from typing import List, Dict, Tuple, Hashable

from manim import ( Scene, Graph,
                    BLUE, RED, GREEN, YELLOW,
                    Create, FadeOut, ManimColor,
                    tempconfig
)
import numpy as np

rng = np.random.default_rng()

class DynamicGraph(Scene):
    def construct(self):
        
        n_nodes: int = 10
        num_states = 10

        nodes: List[Hashable] = [str(i) for i in range(1, n_nodes+1)]
        adjacency_matrix: np.ndarray = rng.integers(0, 2, (n_nodes, n_nodes))
        adjacency_matrix[np.diag_indices(n_nodes)] = 0
        laplacian_matrix: np.ndarray = np.diag(np.sum(adjacency_matrix, axis=1)) - adjacency_matrix
        n_edges: int = (adjacency_matrix.shape[0]*adjacency_matrix.shape[1]) - np.sum(adjacency_matrix == 0)
        # edge_list: List[Tuple[Hashable, Hashable]] = [None for _ in range(n_edges)]
        edge_list: List[Tuple[Hashable, Hashable]] = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adjacency_matrix[i, j] == 1:
                    edge_list.append((nodes[i], nodes[j]))

        # Create a simple graph
        graph = Graph(
            nodes,
            edge_list,

            layout="spring",  # Layout for positioning nodes
        )

        # Display the initial graph
        self.play(Create(graph))
        self.wait(1)


        # Sample color cube in most distal order
        rgbs: np.ndarray = rng.uniform(0.0, 1.0, (n_nodes, 4))
        colors: List[ManimColor] = [None for _ in range(n_nodes)]

        for i in range(n_nodes):
            colors[i] = ManimColor.from_rgb((rgbs[i, 0], rgbs[i, 1], rgbs[i, 2], rgbs[i, 3]))


        states: List[Dict[Hashable, ManimColor]] = [{} for _ in range(num_states)]
        states[0] = {nodes[i]: colors[i] for i in range(n_nodes)}

        for i in range(1,num_states):

            # state: Dict[Hashable, ManimColor] = {}
            state: Dict[Hashable, ManimColor] = states[i]
            state_num = rgbs[:,:3] + iterate_laplacian(laplacian_matrix, rgbs[:,:3], delta=0.5)

            static_idx = rng.integers(0, n_nodes)
            state_num[static_idx] = rgbs[static_idx][:3]
            # state_num = rng.uniform(0.0, 1.0, (n_nodes, 4))
            for j in range(n_nodes):
                state[nodes[j]] = ManimColor.from_rgb(state_num[j].tolist())
            # states.append(state)
            states[i] = state

        # Animate the graph through the states
        # for state in states:
        for time_step in range(num_states):
            state = states[time_step]
            animations = []
            for node, color in state.items():
                animations.append(graph[node].animate.set_fill(color))
            self.play(*animations)
            self.wait(0.5)

        # Conclude
        self.play(FadeOut(graph))

    # def get_state(self,)

def iterate_laplacian(laplacian_matrix: np.ndarray,
                      state: np.ndarray, delta:float = 1.0) -> np.ndarray:
    return delta * laplacian_matrix @ state

with tempconfig({"quality": "medium_quality", "disable_caching": True}):
    scene = DynamicGraph()
    scene.render()


