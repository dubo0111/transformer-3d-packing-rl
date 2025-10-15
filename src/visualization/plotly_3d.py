"""
3D Visualization using Plotly

Interactive 3D visualization of bin packing solutions with support for
animation, height maps, and multiple view modes.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from ..environment.container import Container, PackedItem


class PackingVisualizer:
    """
    Interactive 3D visualization for bin packing.

    Features:
    - Render packed boxes as 3D meshes
    - Interactive rotation, zoom, pan
    - Color-coded boxes with labels
    - Container boundary visualization
    - Height map heatmap view
    - Animation of packing sequence
    - Export to HTML for sharing

    Example:
        >>> visualizer = PackingVisualizer()
        >>> visualizer.visualize_container(container)
        >>> visualizer.save_html("packing_result.html")
    """

    def __init__(self, color_scheme: str = "Viridis"):
        """
        Initialize visualizer.

        Args:
            color_scheme: Plotly color scheme for boxes
        """
        self.color_scheme = color_scheme
        self.fig = None

    def visualize_container(
        self,
        container: Container,
        show_height_map: bool = False,
        show_container_bounds: bool = True,
        show_labels: bool = True,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Visualize packed container in 3D.

        Args:
            container: Container with packed items
            show_height_map: Whether to show height map overlay
            show_container_bounds: Whether to show container boundaries
            show_labels: Whether to show item labels
            title: Plot title

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Generate colors for items
        n_items = len(container.packed_items)
        colors = px.colors.sample_colorscale(
            self.color_scheme, [i / max(n_items, 1) for i in range(n_items)]
        )

        # Add packed items
        for idx, packed_item in enumerate(container.packed_items):
            self._add_box(
                fig,
                position=packed_item.position,
                dimensions=packed_item.dimensions,
                color=colors[idx] if idx < len(colors) else "blue",
                name=f"Item {packed_item.item_id}",
                show_label=show_labels,
            )

        # Add container bounds
        if show_container_bounds:
            self._add_container_bounds(fig, container)

        # Configure layout
        if title is None:
            title = f"3D Bin Packing (Utilization: {container.utilization:.1%})"

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title="Length", range=[0, container.length]),
                yaxis=dict(title="Width", range=[0, container.width]),
                zaxis=dict(title="Height", range=[0, container.height]),
                aspectmode="data",
            ),
            showlegend=True,
            hovermode="closest",
        )

        self.fig = fig
        return fig

    def _add_box(
        self,
        fig: go.Figure,
        position: Tuple[float, float, float],
        dimensions: Tuple[float, float, float],
        color: str,
        name: str,
        show_label: bool = True,
        opacity: float = 0.8,
    ):
        """
        Add a 3D box to the figure.

        Args:
            fig: Plotly figure
            position: (x, y, z) bottom-left-front corner
            dimensions: (length, width, height)
            color: Box color
            name: Box name/label
            show_label: Whether to show label
            opacity: Box opacity
        """
        x, y, z = position
        l, w, h = dimensions

        # Define vertices of the box
        vertices = np.array([
            [x, y, z],          # 0: bottom-left-front
            [x + l, y, z],      # 1: bottom-right-front
            [x + l, y + w, z],  # 2: bottom-right-back
            [x, y + w, z],      # 3: bottom-left-back
            [x, y, z + h],      # 4: top-left-front
            [x + l, y, z + h],  # 5: top-right-front
            [x + l, y + w, z + h],  # 6: top-right-back
            [x, y + w, z + h],  # 7: top-left-back
        ])

        # Define the 12 triangular faces (2 triangles per face, 6 faces)
        # Each face defined by indices into vertices array
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

        # Create mesh
        mesh = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            name=name,
            hovertext=f"{name}<br>Pos: ({x:.2f}, {y:.2f}, {z:.2f})<br>Dim: ({l:.2f}, {w:.2f}, {h:.2f})",
            hoverinfo="text",
            showlegend=True,
        )

        fig.add_trace(mesh)

    def _add_container_bounds(self, fig: go.Figure, container: Container):
        """
        Add container boundary wireframe.

        Args:
            fig: Plotly figure
            container: Container object
        """
        L, W, H = container.length, container.width, container.height

        # Define edges of the container
        edges = [
            # Bottom face
            ([0, L], [0, 0], [0, 0]),
            ([L, L], [0, W], [0, 0]),
            ([L, 0], [W, W], [0, 0]),
            ([0, 0], [W, 0], [0, 0]),
            # Top face
            ([0, L], [0, 0], [H, H]),
            ([L, L], [0, W], [H, H]),
            ([L, 0], [W, W], [H, H]),
            ([0, 0], [W, 0], [H, H]),
            # Vertical edges
            ([0, 0], [0, 0], [0, H]),
            ([L, L], [0, 0], [0, H]),
            ([L, L], [W, W], [0, H]),
            ([0, 0], [W, W], [0, H]),
        ]

        for x, y, z in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    def visualize_height_map(
        self,
        container: Container,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Visualize height map as 2D heatmap.

        Args:
            container: Container with height map
            title: Plot title

        Returns:
            Plotly figure
        """
        height_map = container.height_map

        if title is None:
            title = f"Height Map (Max: {height_map.max():.2f})"

        fig = go.Figure(
            data=go.Heatmap(
                z=height_map,
                colorscale="Viridis",
                colorbar=dict(title="Height"),
            )
        )

        fig.update_layout(
            title=title,
            xaxis=dict(title="Grid X"),
            yaxis=dict(title="Grid Y"),
        )

        return fig

    def create_animation(
        self,
        containers: List[Container],
        frame_duration: int = 500,
        title: str = "Packing Animation",
    ) -> go.Figure:
        """
        Create animation of packing sequence.

        Args:
            containers: List of container states (snapshots)
            frame_duration: Duration of each frame in ms
            title: Animation title

        Returns:
            Plotly figure with animation
        """
        # Create frames
        frames = []
        for i, container in enumerate(containers):
            fig = self.visualize_container(
                container,
                show_container_bounds=True,
                title=f"Step {i + 1}/{len(containers)}",
            )
            frames.append(go.Frame(data=fig.data, name=str(i)))

        # Create initial figure
        initial_fig = self.visualize_container(containers[0], title=title)

        # Add frames
        initial_fig.frames = frames

        # Add animation controls
        initial_fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=frame_duration, redraw=True),
                                    fromcurrent=True,
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(i)],
                                dict(
                                    frame=dict(duration=frame_duration, redraw=True),
                                    mode="immediate",
                                ),
                            ],
                            label=str(i + 1),
                        )
                        for i in range(len(containers))
                    ],
                    active=0,
                    transition=dict(duration=0),
                    x=0.1,
                    y=0,
                    currentvalue=dict(
                        prefix="Step: ",
                        visible=True,
                        xanchor="center",
                    ),
                    len=0.9,
                )
            ],
        )

        return initial_fig

    def save_html(self, filepath: str):
        """
        Save current figure as HTML.

        Args:
            filepath: Path to save HTML file
        """
        if self.fig is None:
            raise ValueError("No figure to save. Call visualize_container first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.fig.write_html(str(filepath))
        print(f"Visualization saved to: {filepath}")

    def show(self):
        """Display current figure."""
        if self.fig is None:
            raise ValueError("No figure to show. Call visualize_container first.")

        self.fig.show()

    @staticmethod
    def compare_solutions(
        containers: List[Container],
        titles: Optional[List[str]] = None,
        rows: int = 1,
    ) -> go.Figure:
        """
        Compare multiple packing solutions side-by-side.

        Args:
            containers: List of containers to compare
            titles: List of titles for each subplot
            rows: Number of rows in subplot grid

        Returns:
            Plotly figure with subplots
        """
        from plotly.subplots import make_subplots

        n_containers = len(containers)
        cols = (n_containers + rows - 1) // rows

        if titles is None:
            titles = [f"Solution {i + 1}" for i in range(n_containers)]

        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "scene"}] * cols for _ in range(rows)],
            subplot_titles=titles,
        )

        visualizer = PackingVisualizer()

        for idx, container in enumerate(containers):
            row = idx // cols + 1
            col = idx % cols + 1

            # Generate visualization for this container
            temp_fig = visualizer.visualize_container(container)

            # Add traces to subplot
            for trace in temp_fig.data:
                fig.add_trace(trace, row=row, col=col)

        fig.update_layout(
            title="Packing Solutions Comparison",
            showlegend=False,
        )

        return fig
