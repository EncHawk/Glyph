from manim import *
import numpy as np

class NeuralNetworkIllustration(Scene):
    def construct(self):
        # Configuration
        layer_sizes = [3, 8, 8, 1]
        layer_spacing = 2.5
        neuron_radius = 0.15
        neuron_color = BLUE_D
        edge_color = GREY_E
        forward_color = YELLOW
        backward_color = RED
        
        # Title setup
        title = Text("Neural Network Architecture").to_edge(UP)
        self.add(title)

        # Create Layers and Neurons
        layers = VGroup()
        for i, size in enumerate(layer_sizes):
            layer = VGroup()
            for j in range(size):
                neuron = Circle(radius=neuron_radius, color=neuron_color, fill_opacity=1)
                neuron.move_to(
                    [(i - len(layer_sizes) / 2) * layer_spacing, 
                     (j - size / 2) * 0.5, 0]
                )
                layer.add(neuron)
            layers.add(layer)

        # Create Edges
        edge_groups = VGroup()
        for i in range(len(layers) - 1):
            edges = VGroup()
            for n1 in layers[i]:
                for n2 in layers[i+1]:
                    edge = Line(n1.get_center(), n2.get_center(), stroke_width=1, color=edge_color)
                    edges.add(edge)
            edge_groups.add(edges)

        # Draw Network
        self.play(
            LaggedStart(*[Create(layer) for layer in layers], lag_ratio=0.3),
            LaggedStart(*[Create(group) for group in edge_groups], lag_ratio=0.3),
            run_time=3
        )
        self.wait(1)

        # Forward Propagation Step
        new_title = Text("Forward Propagation").to_edge(UP)
        self.play(Transform(title, new_title))

        for i in range(len(edge_groups)):
            animations = []
            for edge in edge_groups[i]:
                animations.append(ShowPassingFlash(
                    edge.copy().set_color(forward_color).set_stroke(width=3),
                    time_width=0.5
                ))
            self.play(AnimationGroup(*animations), run_time=1)
            self.play(layers[i+1].animate.set_fill(forward_color), run_time=0.2)
            self.play(layers[i+1].animate.set_fill(neuron_color), run_time=0.2)

        self.wait(1)

        # Backpropagation Step
        new_title = Text("Backpropagation (Error Gradient)").to_edge(UP)
        self.play(Transform(title, new_title))

        for i in reversed(range(len(edge_groups))):
            animations = []
            # Invert line direction for backward visual flow
            for edge in edge_groups[i]:
                back_edge = Line(edge.get_end(), edge.get_start(), stroke_width=3, color=backward_color)
                animations.append(ShowPassingFlash(
                    back_edge,
                    time_width=0.5
                ))
            self.play(AnimationGroup(*animations), run_time=1)
            self.play(layers[i].animate.set_fill(backward_color), run_time=0.2)
            self.play(layers[i].animate.set_fill(neuron_color), run_time=0.2)

        self.wait(2)

        # Final state
        final_title = Text("Training Complete").to_edge(UP)
        self.play(Transform(title, final_title))
        self.play(layers.animate.set_color(GREEN), edge_groups.animate.set_color(GREEN_E))
        self.wait(2)

class LinearRegressionExplanation(Scene):
    def construct(self):
        # Configuration
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"include_tip": False}
        ).shift(DOWN * 0.5)
        
        # Data Points
        x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_data = np.array([1.5, 3.8, 3.2, 5.5, 4.8, 7.2, 6.5, 8.8, 8.2])
        dots = VGroup(*[Dot(axes.c2p(x, y), color=BLUE) for x, y in zip(x_data, y_data)])
        
        # Titles
        title = Title("1. Data Collection")
        
        self.add(title)
        self.play(Create(axes), run_time=2)
        self.play(Create(dots), run_time=3)
        self.wait(0.1)

        # Hypothesis Function
        new_title = Title("2. Hypothesis: y = mx + c")
        m = ValueTracker(0.2)
        c = ValueTracker(1.0)
        
        line = always_redraw(lambda: axes.plot(
            lambda x: m.get_value() * x + c.get_value(),
            color=YELLOW,
            x_range=[0, 10]
        ))
        
        self.play(Transform(title, new_title))
        self.play(Create(line), run_time=2)
        self.play(m.animate.set_value(0.5), c.animate.set_value(2.0), run_time=2.9)

        # Residuals
        new_title = Title("3. Calculating Residuals (Error)")
        
        def get_residuals():
            res = VGroup()
            for x, y in zip(x_data, y_data):
                p1 = axes.c2p(x, y)
                p2 = axes.c2p(x, m.get_value() * x + c.get_value())
                res.add(DashedLine(p1, p2, color=RED, stroke_width=2))
            return res

        residuals = always_redraw(get_residuals)
        
        self.play(Transform(title, new_title))
        self.play(Create(residuals), run_time=3)
        self.wait(1.9)

        # Optimization
        new_title = Title("4. Minimizing Loss (Gradient Descent)")
        
        self.play(Transform(title, new_title))
        self.play(
            m.animate.set_value(0.9),
            c.animate.set_value(0.7),
            run_time=4.9,
            rate_func=bezier([0, 0, 1, 1])
        )

        # Final Fit
        new_title = Title("Result: Optimal Linear Fit")
        equation = MathTex("y = 0.9x + 0.7", color=YELLOW).to_edge(RIGHT).shift(UP)
        
        self.play(Transform(title, new_title))
        self.play(Write(equation), run_time=2)
        self.play(FadeOut(residuals), run_time=2)
        self.play(Indicate(line), run_time=0.9)
        self.wait(0.1)


if __name__ == "__main__":
    dr = LinearRegressionExplanation()
    dr.construct()
