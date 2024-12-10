from typing import List, Dict, Tuple, Hashable

from manim import ( Scene, Graph,
                    BLUE, RED, GREEN, YELLOW,
                    Create, FadeOut, FadeIn, FadeTransform,
                    ManimColor,
                    tempconfig, Mobject,
                    DEGREES, MarkupText, UP, DOWN, LEFT, RIGHT,
                    always_redraw,
)
from manim.opengl import OpenGLSurface, OpenGLSurfaceMesh
import numpy as np

from utils import surface_height

rng = np.random.default_rng()

class DynamicGraph(Scene):

    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.time_step: int = 0
        self.mobjects_dict: Dict[(str, Mobject)] = {}

    def add_object(self, name:str, mobject: Mobject):
        if not isinstance(name, str):
            raise ValueError(f"Name must be a string, got {name}:{type(name)}")
        self.mobjects_dict[name] = mobject

    def initialise_graph(self, num_states: int, num_nodes: int):

        self.nodes: List[Hashable] = [str(i) for i in range(1, num_nodes+1)]
        self.adjacency_matrix: np.ndarray = rng.integers(0, 2, (num_nodes, num_nodes))
        self.adjacency_matrix[np.diag_indices(num_nodes)] = 0
        self.laplacian_matrix: np.ndarray = np.diag(np.sum(self.adjacency_matrix, axis=1)) - self.adjacency_matrix
        n_edges: int = (self.adjacency_matrix.shape[0]*self.adjacency_matrix.shape[1]) - np.sum(self.adjacency_matrix == 0)
        # edge_list: List[Tuple[Hashable, Hashable]] = [None for _ in range(n_edges)]
        self.edge_list: List[Tuple[Hashable, Hashable]] = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if self.adjacency_matrix[i, j] == 1:
                    self.edge_list.append((self.nodes[i], self.nodes[j]))

        # Create a simple graph
        graph = Graph(
            self.nodes,
            self.edge_list,
            edge_config={
                "stroke_width": 0.5,
            },
            layout="spring",  # Layout for positioning nodes
        )

        self.add_object(name="graph", mobject=graph)

        # Display the initial graph
        self.play(Create(graph))
        self.wait(1)




    def iterate_laplacian(self, laplacian_matrix: np.ndarray,
                          state: np.ndarray, delta:float = 1.0) -> np.ndarray:
        return -delta * laplacian_matrix @ state

    def compute_gaussian_mixture_surface(self,
            node_positions, precisions, grid_x, grid_y) -> np.ndarray:
        """Compute the mixture of Gaussians on a 2D grid."""
        surface: np.ndarray = np.zeros_like(grid_x)
        # for i, node in enumerate(nodes):
        if node_positions.shape[1] != 2:
            raise ValueError(
                    "Node positions must have 2 dimensions",
                    f"Got {node_positions.shape[1]} dimensions"
            )

        for i, (mu_x, mu_y) in enumerate(node_positions):
            sigma = 1 / np.sqrt(precisions[i])  # Precision to standard deviation
            gaussian = np.exp(-((grid_x - mu_x)**2 + (grid_y - mu_y)**2) / (2 * sigma**2))
            surface += gaussian / (2 * np.pi * sigma**2)
        return surface



    def init_surface(self) -> OpenGLSurface:

        # Define a grid for the lattice
        x_range = np.linspace(-2, 2, 50)
        y_range = np.linspace(-2, 2, 50)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        self.grid_x = grid_x
        self.grid_y = grid_y
        
        # # Compute the Gaussian mixture surface
        surface_values = self.compute_gaussian_mixture_surface(
            self.node_positions, self.precisions, grid_x, grid_y)

        # # Use OpenGLSurface for visualization
        surface = OpenGLSurface(
            lambda u, v: np.array([u, v, surface_height(u, v, grid_x, grid_y, surface_values)]),
            u_range=[x_range[0], x_range[-1]],
            v_range=[y_range[0], y_range[-1]],
            resolution=(50, 50),
        )
        surface.set_color(BLUE)

        # surface_mesh: OpenGLSurfaceMesh = OpenGLSurfaceMesh(surface)
        # self.play(Create(surface_mesh))
        
        self.play(Create(surface))
        # self.play(FadeTransform(surface_mesh, surface))
        self.play(surface.animate.set_opacity(0.3))
        self.wait(0.1)

        self.surface = surface

        return surface

    def update_surface_(self):
        """Update the surface values and refresh the OpenGLSurface object."""
    
        # Recompute the surface values
        surface_values = self.compute_gaussian_mixture_surface(
            self.node_positions, self.precisions, self.grid_x, self.grid_y
        )
    
        # Update the surface
        # self.surface.apply_function(lambda u, v: np.array(
        self.surface.function = lambda u, v: np.array(
            [u, v, surface_height(u, v, self.grid_x, self.grid_y, surface_values)]
        )

        # print("surface values: \n", surface_values)
        print("precisions: \n", self.precisions)

        return self.surface

    def update_surface(self):
        """Recompute and recreate the surface with updated values."""
        # Recompute surface values
        surface_values = self.compute_gaussian_mixture_surface(
            self.node_positions, self.precisions, self.grid_x, self.grid_y
        )
        
        # Create a new surface with updated function
        new_surface = OpenGLSurface(
            lambda u, v: np.array(
                [u, v, surface_height(u, v, self.grid_x, self.grid_y, surface_values)]
            ),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(50, 50),
        )
        new_surface.set_color(BLUE).set_opacity(0.3)
        
        # Replace the old surface with the new one
        self.remove(self.surface)
        self.add(new_surface)
        self.surface = new_surface

    def construct(self):

        num_nodes: int = 6
        num_states = 20

        self.initialise_graph(
            num_states=num_states,
            num_nodes=num_nodes
        )
        # Node positions and precisions

        graph=self.mobjects_dict["graph"]
        self.node_positions = np.concatenate(
            [[graph[node].get_coord(dim=[0,1]) for node in self.nodes]], axis=0)
         

        ### Initialise states and colors

        # Sample color cube in most distal order
        rgbs: np.ndarray = rng.uniform(0.0, 1.0, (num_nodes, 4))
        rgbs[:, 3] = 1.0
        node_colors: List[ManimColor] = [None for _ in range(num_nodes)]

        for i in range(num_nodes):
            node_colors[i] = ManimColor.from_rgb((rgbs[i, 0], rgbs[i, 1], rgbs[i, 2], rgbs[i, 3]))


        states: List[Dict[Hashable, ManimColor]] = [{} for _ in range(num_states)]
        states[0] = {self.nodes[i]: node_colors[i] for i in range(num_nodes)}
        states_num: np.ndarray = np.zeros((num_states, num_nodes, 3))

        # Initialise surface
        self.precisions: np.ndarray = np.diag(rgbs@rgbs.T).flatten()**2
        # self.precisions: np.ndarray = np.random.uniform(0.1, 8., num_nodes)
        assert self.precisions.shape[0] == num_nodes, f"precisions.shape[0]: {self.precisions.shape[0]}, num_nodes: {num_nodes}"

        self.init_surface()
        # always_redraw(self.update_surface)
        self.wait()

        # Tilt camera down a bit
        self.play(self.camera.animate.set_euler_angles(theta=-10*DEGREES, phi=40*DEGREES))

        ### Iterate through the states

        delta = 0.1
        for i in range(1,num_states):

            state: Dict[Hashable, ManimColor] = states[i]
            state_num = rgbs[:,:3] + self.iterate_laplacian(self.laplacian_matrix, rgbs[:,:3], delta=delta)
            states_num[i] = state_num
            rgbs[:,:3] = state_num

            for j in range(num_nodes):
                state[self.nodes[j]] = ManimColor.from_rgb(state_num[j].tolist())
            # states.append(state)
            states[i] = state

        # Animate the graph through the states
        # light = self.camera.light_source
        # time_text = MarkupText("<u>time: 0</u>").next_to(
        #         self.mobjects_dict["graph"], UP
        # )
        # self.play(FadeIn(time_text))
        for time_step in range(num_states):
            # self.play(light.animate.shift([0, 0, 1]))
            
            state = states[time_step]
            animations = [None for _ in range(num_nodes)]
            for node_idx, (node, color) in enumerate(state.items()):
                animations[node_idx] = self.mobjects_dict["graph"][node].animate.set_fill(color)

            # time_text.become(MarkupText(f"<u>time: {time_step}</u>").next_to(
            #     self.mobjects_dict["graph"], UP
            # ))
            self.play(*animations)
            # self.play(FadeTransform(time_text, time_text))

            # Update precisions
            self.precisions[:] = np.diag(states_num[time_step] @ states_num[time_step].T).flatten()**2
            # self.precisions[:] = np.random.uniform(0.1, 0.8, num_nodes)
            self.update_surface()
            self.wait()
    
        # Conclude
        self.play(FadeOut(self.mobjects_dict["graph"]))
