from typing import List, Dict, Tuple, Hashable

from manim import ( Scene, Graph,
                    BLUE, RED, GREEN, YELLOW,
                    Create, FadeOut, ManimColor,
                    tempconfig,
                    Text, Write,
                    DEGREES, FadeTransform
)
from manim.opengl import *
import numpy as np

rng = np.random.default_rng()

class Intro(Scene):

    def construct(self):

        if False:
            text = Text("Hello, World!")
            self.play(Write(text))
            # self.play(
            #     self.camera.animate.set_euler_angle(
            #         theta=-10*DEGREES,
            #         phi=70*DEGREES,
            #     )
            # )

            self.play(FadeOut(text))

        surface: OpenGLSurface = OpenGLSurface(
            lambda u, v: np.array([u, v, u*np.sin(v) + v*np.cos(u)]),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(20, 20),
        )
        surface_mesh: OpenGLSurfaceMesh = OpenGLSurfaceMesh(surface)
        self.play(Create(surface_mesh))
        self.play(FadeTransform(surface_mesh, surface))
        self.wait()


if __name__ == "__main__":
    with tempconfig({
        "quality": "medium_quality",
        "disable_caching": True,
        # "config.renderer": "opengl",
    }):
        scene = Intro()
        scene.render()
