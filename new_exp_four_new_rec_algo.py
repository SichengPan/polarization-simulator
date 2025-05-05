import os
import dash
import json
import threading
import webbrowser
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.colors
from dash import dcc, html
from dash.dependencies import Input, Output, State
from base_code import SocialNetworkModel  # Import the simulation model

# === Initialize Dash App ===
app = dash.Dash(__name__)

# === Default Experiment Parameters ===
DEFAULT_N = 100  
DEFAULT_P = 0.01  
DEFAULT_NUM_ROUNDS = 30 
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_DROPOUT = 5 
seed = 42  

# === Initialize Social Network Model ===
model = SocialNetworkModel(N=DEFAULT_N, p=DEFAULT_P, network_type="random", dynamic_edges=True, seed=seed)
all_experiment_data = []  

# Prevent multiple browser tabs from opening
browser_opened = False

# Global state variables
current_round = 0  
polarization_values = [] 
radicalization_values = []
average_degree_values = []
echo_chamber_values = [] 
modularity_values = []
processing = False  # Global processing flag


def get_network_figure(G):
    """ 
    Generate a network visualization using Plotly. 
    
    This function takes a NetworkX graph `G` and creates a Plotly figure displaying:
    - Nodes as blue markers
    - Edges as gray lines
    - Hover information, including node ID, opinion vector, attribute vector (if available), and neighbors with similarity scores.
    
    Parameters:
    - G (networkx.Graph): The input social network graph.
    
    Returns:
    - fig (plotly.graph_objects.Figure): A Plotly figure object representing the network.
    """
    pos = nx.spring_layout(G, seed=42)  
    edge_x, edge_y = [], []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        mode="lines"
    )

    node_x, node_y, node_id_data, hovertext = [], [], [], []

    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        node_id_data.append(node_id)
        
        # Get the node
        node = model.nodes.get(node_id)
        if node:
            opinion_text = f"Opinion: {node.get_opinion()}" if hasattr(node, "get_opinion") else "Opinion: N/A"
            attribute_text = f"Attributes: {node.get_attributes()}" if hasattr(node, "get_attributes") else "Attributes: N/A"

            # similarities with neighbors
            neighbors_similarity = model.calculate_all_neighbors_similarity(node_id, alpha=1.0)
            neighbor_text = "<br>".join([f"Neighbor {n}: {s:.2f}" for n, s in neighbors_similarity.items()])

            # multiple lines
            hovertext.append(f"Node {node_id}<br>{opinion_text}<br>{attribute_text}<br>Neighbors (and Similarity):<br>{neighbor_text}")
        else:
            hovertext.append(f"Node {node_id}<br>(No Data Available)")


    # node_trace to show key info 
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(size=8, color="blue"),
        customdata=node_id_data,  # store node_id_data for later use
        hoverinfo="text",
        hovertext=hovertext,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


# === Dash Layout ===
app.layout = html.Div([
    html.H1("Polarization in Social Networks --- Experiment Dashboards"),

    # === Base experiment parameters ===
    html.Label("Number of Nodes (N):"),
    dcc.Input(id="input-n", type="number", value=DEFAULT_N, min=10, step=10),
    
    html.Label("Initial Edge Probability (p):"),
    dcc.Input(id="input-p", type="number", value=DEFAULT_P, min=0, max=1, step=0.001),
    
    html.Label("Number of Rounds (num_rounds):"),
    dcc.Input(id="input-num-rounds", type="number", value=DEFAULT_NUM_ROUNDS, min=10, step=10),
    html.Br(),
    html.Br(),

    # === Parameters for Connection / Disconnection ===
    html.Label("Threshold for Connecting Edges:"),
    dcc.Input(id="input-threshold-connect", type="number", value=0.75, min=0, max=1, step=0.01),

    html.Label("Threshold for Disconnecting Edges:"),
    dcc.Input(id="input-threshold-disconnect", type="number", value=0.25, min=0, max=1, step=0.01),
    html.Br(),
    html.Br(),


    # === Attribute Vectors === 
    html.H2("Attribute Vector Settings"),

    html.Div([
        html.Label("Enable Attribute Vector:", style={"marginRight": "10px"}),
        dcc.Checklist(
            id="enable-attribute-vector",
            options=[{"label": "Enable", "value": "enabled"}],
            value=[],
            style={"display": "inline-block"}
        ),
    ], style={"display": "flex", "alignItems": "center"}),
    html.Br(),

    html.Label("Attribute Vector Length:"),
    dcc.Input(
        id="attribute-vector-length",
        type="number",
        value=1,
        min=1,
        step=1,
    ),
    html.Br(),
    html.Br(),

    html.Label("Alpha User (0-1):"),
    dcc.Slider(
        id="alpha-user",
        min=0, max=1, step=0.01, value=1,
        marks={0: "0", 0.5: "0.5", 1: "1"}
    ),
    html.Div(id="alpha-user-display", style={"textAlign": "center", "marginTop": "10px"}),

    html.Br(),



    # === Recommendation Algorithm Selections ===
    html.H2("Recommendation Algorithm Settings"),

    # Content-Based Recommendation
    html.Div([
        html.Label("Content-Based Recommendation", style={"display": "inline-block", "font-weight": "bold"}),
        dcc.Checklist(id="enable-content-based", 
                    options=[{"label": "Use This Algorithm", "value": "enabled"}], 
                    value=[], 
                    style={"display": "inline-block", "margin-left": "10px"}),
        dcc.Input(id="num-content-based", type="number", value=5, min=1, step=1, 
                style={"width": "50px", "display": "inline-block", "margin-left": "10px"})
    ], style={"display": "flex", "align-items": "center", "margin-bottom": "10px"}),

    # Dissimilar Recommendation
    html.Div([
        html.Label("Dissimilar Recommendation", style={"display": "inline-block", "font-weight": "bold"}),
        dcc.Checklist(id="enable-dissimilar", 
                    options=[{"label": "Use This Algorithm", "value": "enabled"}], 
                    value=[], 
                    style={"display": "inline-block", "margin-left": "10px"}),
        dcc.Input(id="num-dissimilar", type="number", value=5, min=1, step=1, 
                style={"width": "50px", "display": "inline-block", "margin-left": "10px"})
    ], style={"display": "flex", "align-items": "center", "margin-bottom": "10px"}),

    # Collaborative Filtering Recommendation
    html.Div([
        html.Label("Collaborative Filtering Recommendation", style={"display": "inline-block", "font-weight": "bold"}),
        dcc.Checklist(id="enable-collaborative", 
                    options=[{"label": "Use This Algorithm", "value": "enabled"}], 
                    value=[], 
                    style={"display": "inline-block", "margin-left": "10px"}),
        dcc.Input(id="num-collaborative", type="number", value=5, min=1, step=1, 
                style={"width": "50px", "display": "inline-block", "margin-left": "10px"})
    ], style={"display": "flex", "align-items": "center", "margin-bottom": "10px"}),

    # Random Recommendation
    html.Div([
        html.Label("Random Recommendation", style={"display": "inline-block", "font-weight": "bold"}),
        dcc.Checklist(id="enable-random", 
                    options=[{"label": "Use This Algorithm", "value": "enabled"}], 
                    value=[], 
                    style={"display": "inline-block", "margin-left": "10px"}),
        dcc.Input(id="num-random", type="number", value=5, min=1, step=1, 
                style={"width": "50px", "display": "inline-block", "margin-left": "10px"})
    ], style={"display": "flex", "align-items": "center", "margin-bottom": "10px"}),

    # Attribute Vector Recommendation
    html.Div([
        html.Label("Attribute Vector Recommendation", style={"display": "inline-block", "font-weight": "bold"}),
        dcc.Checklist(id="enable-attribute-vector-recommendation", 
                    options=[{"label": "Use This Algorithm", "value": "enabled"}], 
                    value=[], 
                    style={"display": "inline-block", "margin-left": "10px"}),
        dcc.Input(id="num-attribute-vector-recommendation", type="number", value=5, min=1, step=1, 
                style={"width": "50px", "display": "inline-block", "margin-left": "10px"})
    ], style={"display": "flex", "align-items": "center", "margin-bottom": "10px"}),

    # Hybrid Recommendation
    html.Div([
        html.Label("Weighted Attribute & Opinion Recommendation", style={"display": "inline-block", "font-weight": "bold"}),
        dcc.Checklist(id="enable-hybrid-recommendation", 
                    options=[{"label": "Use This Algorithm", "value": "enabled"}], 
                    value=[], 
                    style={"display": "inline-block", "margin-left": "10px"}),
        dcc.Input(id="num-hybrid-recommendation", type="number", value=5, min=1, step=1, 
                style={"width": "50px", "display": "inline-block", "margin-left": "10px"})
    ], style={"display": "flex", "align-items": "center", "margin-bottom": "10px"}),

    # Attribute & Dissimilar Recommendation
    html.Div([
        html.Label("Dissimilarity & Attribute-Based Recommendation", style={"display": "inline-block", "font-weight": "bold"}),
        dcc.Checklist(id="enable-attribute-dissimilar-recommendation", 
                    options=[{"label": "Use This Algorithm", "value": "enabled"}], 
                    value=[], 
                    style={"display": "inline-block", "margin-left": "10px"}),
        dcc.Input(id="num-attribute-dissimilar-recommendation", type="number", value=5, min=1, step=1, 
                style={"width": "50px", "display": "inline-block", "margin-left": "10px"})
    ], style={"display": "flex", "align-items": "center", "margin-bottom": "10px"}),

    html.Br(),

    # === Alpha Recommendation Slider ===
    html.Label("Alpha Recommendation (0-1):"),
    dcc.Slider(
        id="alpha-recommendation",
        min=0, max=1, step=0.01, value=1,
        marks={0: "0", 0.5: "0.5", 1: "1"}
    ),
    html.Div(id="alpha-recommendation-display", style={"textAlign": "center", "marginTop": "10px"}),

    html.Br(),



    # Sample Size & Dropout
    html.H2("Edge Modification Parameter"),

    # html.Label("Max New Connections Per Node (sample_size_connect):"),
    # dcc.Input(id="input-sample-size-connect", type="number", value=DEFAULT_SAMPLE_SIZE, min=0, step=1),

    # html.Br(),
    # html.Br(),

    html.Label("Dropout (Number of Nodes to Consider Dropping Connections):"),
    dcc.Input(id="input-dropout", type="number", value=DEFAULT_DROPOUT, min=0, step=1),

    html.Br(),
    html.Br(),


    # === Edge Adjustment Period Slider ===
    html.H2("Edge Adjustment Frequency"),

    html.Label("Adjust Edges Every N Rounds:"),
    dcc.Slider(id="edge-period-slider", min=1, max=15, step=1, value=5,
               marks={1: "1", 5: "5", 10: "10"}),
    html.Div(id="slider-value-display", style={"textAlign": "center", "fontSize": "18px", "marginTop": "10px"}),
    html.Div(id="current-round-display", style={"textAlign": "center", "fontSize": "18px", "marginTop": "10px"}),
    html.Br(),


    # === Customize experiment parameter ===
    html.H2("Experiment Configuration & Visualization"),

    html.Label("Enter Changed Experiment Variable (such as 'Edge Adjustment Period = 5'):"),
    dcc.Input(id="experiment-name", type="text", value="Default Experiment", debounce=True),
    html.Br(),
    html.Br(),

    # === Buttons ===
    html.Div(id="button-container", children=[
        html.Button("Next Step", id="next-step-btn", n_clicks=0),
        html.Button("Finish Simulation", id="finish-btn", style={"margin-left": "10px"}),
        html.Button("Reset Simulation", id="reset-btn", style={"margin-left": "10px", "background-color": "#ff5050", "color": "white"}),
        html.Button("Add New Experiment", id="new-experiment-btn", style={"margin-left": "10px", "background-color": "#4CAF50", "color": "white"}),
        html.Button("Export Data", id="export-btn", style={"margin-left": "10px"})
    ]),

    html.Div(id="export-status", style={"margin-top": "10px", "color": "green"}),

    # html.Div(id="node-info", style={"border": "1px solid black", "padding": "10px", "margin-top": "20px"}),
    html.Div(id="node-info", style={"display": "none"}),  # Dash can still find it, but it's hidden

    dcc.Graph(id="network-graph", figure=get_network_figure(model.graph)),  
    dcc.Graph(id="polarization-graph"),
    dcc.Graph(id="radicalization-graph"),
    dcc.Graph(id="average-degree-graph"),
    dcc.Graph(id="echo-chamber-graph"),
    dcc.Graph(id="modularity-graph"),

    dcc.Graph(id="opinion-scatter-plot"),
    dcc.Graph(id="opinion-histogram")
])


# === Callback: Display Nodes ===
@app.callback(
    Output("node-info", "children"),
    [Input("network-graph", "hoverData")],
    [State("alpha-user", "value")]  # Get the value of the alpha-user slider
)
def display_node_info(hoverData, alpha_user):
    """ 
    Display node information when hovering over a node in the network graph.

    This function is triggered when the user hovers over a node in the graph. It retrieves:
    - The node's ID
    - Its opinion vector
    - Its attribute vector (if available)
    - The similarity scores with its neighbors (calculated using the alpha-user value from the UI)
    
    Parameters:
    - hoverData (dict): Data from the hover event, containing information about the node.
    - alpha_user (float): The alpha parameter used to compute similarity with neighbors.

    Returns:
    - list: A list of HTML components displaying node details (or an error message if the node is not found).
    """
    # Validate hoverData to ensure a node is being hovered over
    if not hoverData or "points" not in hoverData or "customdata" not in hoverData["points"][0]:
        return "Hover over a node to see details."

    node_id = hoverData["points"][0]["customdata"]
    print(f"Displaying info for Node {node_id} with alpha = {alpha_user}")

    # Retrieve the Node instance
    node = model.nodes.get(node_id)

    if not node:
        return f"Node {node_id} not found."

    # Opinion Vector
    opinion_vector = node.get_opinion() if hasattr(node, "get_opinion") else "N/A"
    if isinstance(opinion_vector, np.ndarray):
        opinion_vector = np.array2string(opinion_vector)

    # Attribute Vector (if exists)
    attribute_vector = node.get_attributes() if hasattr(node, "get_attributes") else "N/A"
    if isinstance(attribute_vector, np.ndarray):
        attribute_vector = np.array2string(attribute_vector)

    # neighbourhood similarity
    neighbors_similarity = model.calculate_all_neighbors_similarity(node_id, alpha=alpha_user)
    print("Alpha = %s" % alpha_user)

    # Construct the display content
    node_info = [
        html.H4(f"Node {node_id}"),
        html.P(f"Opinion Vector: {opinion_vector}"),
    ]

    if attribute_vector != "N/A":
        node_info.append(html.P(f"Attribute Vector: {attribute_vector}"))

    if neighbors_similarity:
        node_info.append(html.H5("Neighbors & Similarity"))
        node_info.extend([html.P(f"Neighbor {n}: Similarity {s:.2f}") for n, s in neighbors_similarity.items()])
    
    return node_info


# === Callback: Export Data ===
@app.callback(
    Output("export-status", "children"),
    [Input("export-btn", "n_clicks")],
    prevent_initial_call=True
)
def trigger_export(n_clicks):
    print(f"📤 Export button clicked {n_clicks} times")
    if n_clicks:
        excel_path, json_path = export_experiment_data(all_experiment_data)
        if excel_path and json_path:
            return f"Data exported to folder: `{os.path.dirname(excel_path)}`"
        else:
            return "No experiment data available to export."


# === Callback: Update Slider Value Display ===
@app.callback(
    Output("slider-value-display", "children"),
    [Input("edge-period-slider", "value")]
)
def update_slider_value(value):
    """ Display the current edge adjustment period value """
    return f"Current Edge Adjustment Period: {value}"

# === Callback: Update Alpha Slider Value Display ===
@app.callback(
    Output("alpha-user-display", "children"),
    [Input("alpha-user", "value")]
)
def update_alpha_display(value):
    return [
        f"Opinion Similarity Weight for Connection / Disconnection (α): {value * 100:.0f}%",
        html.Br(),
        "100% → Only Opinion Similarity",
        html.Br(),
        "0% → Only Attribute Similarity"
    ]

@app.callback(
    Output("alpha-recommendation-display", "children"),
    [Input("alpha-recommendation", "value")]
)
def update_alpha_recommendation_display(value):
    return [
        f"Recommendation Similarity Weight (α): {value * 100:.0f}%",
        html.Br(),
        "100% → Only Opinion Similarity",
        html.Br(),
        "0% → Only Attribute Similarity"
    ]

# === Callback: Manage Button States ===
@app.callback(
    Output("button-container", "children"),
    [Input("next-step-btn", "n_clicks"),
     Input("finish-btn", "n_clicks"),
     Input("reset-btn", "n_clicks"),
     Input("new-experiment-btn", "n_clicks"),
     Input("export-btn", "n_clicks"),
     Input("network-graph", "figure")],
    prevent_initial_call=True
)
def manage_buttons(next_clicks, finish_clicks, reset_clicks, new_experiment_clicks, export_clicks, graph):
    """ Disable buttons while processing updates, re-enable after completion """
    global processing
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # If simulation is running, disable buttons
    if processing:
        return [
            html.Button("Next Step", id="next-step-btn", disabled=True),
            html.Button("Finish Simulation", id="finish-btn", style={"margin-left": "10px"}, disabled=True),
            html.Button("Reset Simulation", id="reset-btn", style={"margin-left": "10px", "background-color": "#ff5050", "color": "white"}, disabled=True),
            html.Button("Add New Experiment", id="new-experiment-btn", style={"margin-left": "10px", "background-color": "#4CAF50", "color": "white"}, disabled=True),
            html.Button("Export Data", id="export-btn", style={"margin-left": "10px", "background-color": "#008CBA", "color": "white", "display": "inline-block"})
        ]

    # If simulation completed, re-enable buttons
    return [
        html.Button("Next Step", id="next-step-btn", disabled=False),
        html.Button("Finish Simulation", id="finish-btn", style={"margin-left": "10px"}, disabled=False),
        html.Button("Reset Simulation", id="reset-btn", style={"margin-left": "10px", "background-color": "#ff5050", "color": "white"}, disabled=False),
        html.Button("Add New Experiment", id="new-experiment-btn", style={"margin-left": "10px", "background-color": "#4CAF50", "color": "white"}, disabled=False),
        html.Button("Export Data", id="export-btn", style={"margin-left": "10px", "background-color": "#008CBA", "color": "white", "display": "inline-block"})
    ]


@app.callback(
    Output("next-step-btn", "disabled"),
    [Input("current-round-display", "children")]
)
def disable_next_step(current_round_text):
    """ Disable 'Next Step' button when simulation is complete """
    current_round = int(current_round_text.split("/")[0].split(":")[-1].strip())  
    total_rounds = int(current_round_text.split("/")[1].strip())  
    return current_round >= total_rounds  


# === Callback: Run Propagation ===
@app.callback(
    [Output("network-graph", "figure"),
     Output("polarization-graph", "figure"),
     Output("radicalization-graph", "figure"),
     Output("average-degree-graph", "figure"),
     Output("echo-chamber-graph", "figure"),
     Output("modularity-graph", "figure"),
     Output("opinion-scatter-plot", "figure"),
     Output("opinion-histogram", "figure"),

     Output("current-round-display", "children")],
    [Input("next-step-btn", "n_clicks"),
     Input("finish-btn", "n_clicks"),
     Input("reset-btn", "n_clicks"),
     Input("new-experiment-btn", "n_clicks")],
    [State("edge-period-slider", "value"),
     State("input-n", "value"),
     State("input-p", "value"),
     State("input-num-rounds", "value"),
     State("polarization-graph", "figure"),
     State("radicalization-graph", "figure"),
     State("average-degree-graph", "figure"),
     State("echo-chamber-graph", "figure"),
     State("modularity-graph", "figure"),
     State("input-threshold-connect", "value"),
     State("input-threshold-disconnect", "value"),
     State("experiment-name", "value"),
     State("enable-attribute-vector", "value"),
     State("attribute-vector-length", "value"),
     State("alpha-user", "value"),
     State("alpha-recommendation", "value"),
     
     State("enable-content-based", "value"),
     State("num-content-based", "value"),
     State("enable-dissimilar", "value"),
     State("num-dissimilar", "value"),
     State("enable-collaborative", "value"),
     State("num-collaborative", "value"),
     State("enable-random", "value"),
     State("num-random", "value"),
     State("enable-attribute-vector-recommendation", "value"),
     State("num-attribute-vector-recommendation", "value"),
     State("enable-hybrid-recommendation", "value"),
     State("num-hybrid-recommendation", "value"),
     State("enable-attribute-dissimilar-recommendation", "value"),
     State("num-attribute-dissimilar-recommendation", "value"),
     
     State("input-dropout", "value")],
)
def update_simulation(next_clicks, finish_clicks, reset_clicks, new_experiment_clicks, 
                      adjust_edges_every, new_n, new_p, new_num_rounds, 
                      existing_pol_figure, existing_rad_figure, 
                      existing_avg_degree_figure, existing_echo_chamber_figure,
                      existing_modularity_figure,
                      threshold_connect, threshold_disconnect, 
                      experiment_name, 
                      enable_attribute_vector, attribute_vector_length, 
                      alpha_user,
                      alpha_recommendation,

                      content_enabled, content_num,
                      dissimilar_enabled, dissimilar_num,
                      collaborative_enabled, collaborative_num,
                      random_enabled, random_num,

                      attribute_vector_enabled, attribute_vector_num,
                      hybrid_enabled, hybrid_num,
                      attribute_dissimilar_enabled, attribute_dissimilar_num,
                      
                      dropout_size):
    """ Handle Next Step, Finish Simulation, Reset, and Add New Experiment actions """
    global model, current_round, polarization_values, radicalization_values, average_degree_values, echo_chamber_values, modularity_values, all_experiment_data  
    ctx = dash.callback_context  

    if not ctx.triggered:
        return get_network_figure(model.graph), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), get_opinion_scatter_figure(), get_opinion_histogram(), f"Current Round: {current_round} / {new_num_rounds}"  

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # parse recommendation algorithm settings
    use_content_based = "enabled" in content_enabled
    num_content_based = content_num if use_content_based else 0

    use_dissimilar = "enabled" in dissimilar_enabled
    num_dissimilar = dissimilar_num if use_dissimilar else 0

    use_collaborative = "enabled" in collaborative_enabled
    num_collaborative = collaborative_num if use_collaborative else 0

    use_random = "enabled" in random_enabled
    num_random = random_num if use_random else 0

    use_attribute_vector = "enabled" in attribute_vector_enabled
    num_attribute_vector = attribute_vector_num if use_attribute_vector else 0

    use_hybrid = "enabled" in hybrid_enabled
    num_hybrid = hybrid_num if use_hybrid else 0

    use_attribute_dissimilar = "enabled" in attribute_dissimilar_enabled
    num_attribute_dissimilar = attribute_dissimilar_num if use_attribute_dissimilar else 0
    
    
    # parse edge adjustment period
    if adjust_edges_every is None or adjust_edges_every == 0:
        adjust_edges_every = 1

    processing = True


    # Store parameters before this round
    experiment_parameters = {
        "N": new_n,
        "p": new_p,
        "num_rounds": new_num_rounds,
        "threshold_connect": threshold_connect,
        "threshold_disconnect": threshold_disconnect,
        "alpha_user": alpha_user,
        "alpha_recommendation": alpha_recommendation,
        "use_content_based": "enabled" in content_enabled,
        "num_content_based": content_num if "enabled" in content_enabled else 0,
        "use_dissimilar": "enabled" in dissimilar_enabled,
        "num_dissimilar": dissimilar_num if "enabled" in dissimilar_enabled else 0,
        "use_collaborative": "enabled" in collaborative_enabled,
        "num_collaborative": collaborative_num if "enabled" in collaborative_enabled else 0,
        "use_random": "enabled" in random_enabled,
        "num_random": random_num if "enabled" in random_enabled else 0,
        "use_attribute_vector": "enabled" in attribute_vector_enabled,
        "num_attribute_vector": num_attribute_vector if "enabled" in attribute_vector_enabled else 0,
        "use_hybrid": "enabled" in hybrid_enabled,
        "num_hybrid": num_hybrid if "enabled" in hybrid_enabled else 0,
        "use_attribute_dissimilar": "enabled" in attribute_dissimilar_enabled,
        "num_attribute_dissimilar": num_attribute_dissimilar if "enabled" in attribute_dissimilar_enabled else 0,
        "attribute_vector_length": attribute_vector_length if "enabled" in enable_attribute_vector else None
    }

    # experimental buttons & branches
    if button_id == "reset-btn":
        print("🔄 Resetting simulation...")
        model = SocialNetworkModel(N=new_n, p=new_p, network_type="random", dynamic_edges=True, seed=seed)
        current_round = 0
        polarization_values = []
        radicalization_values = []
        average_degree_values = []
        echo_chamber_values = []
        modularity_values = []
        all_experiment_data = []  #  Clear all experiment data
        processing = False

        if "enabled" in enable_attribute_vector:
            print("Add attribute vectors in reset mode, vector length is: ", attribute_vector_length)
            model.initialize_random_attributes(attribute_dim=attribute_vector_length)
        else:
            print("No attribute vectors in reset mode.")

        return get_network_figure(model.graph), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), get_opinion_scatter_figure(), get_opinion_histogram(), f"Current Round: {current_round} / {new_num_rounds}"  

    elif button_id == "next-step-btn":
        if current_round >= new_num_rounds:
            print("🚀 Simulation completed.")
            polarization_values.append(model.measure_polarization())  # ✅ Store final polarization
            radicalization_values.append(model.measure_radicalization())  # ✅ Store radicalization value
            average_degree_values.append(model.compute_average_degree())  # ✅ Store average degree
            echo_chamber_values.append(model.compute_echo_chamber_extent())  # ✅ Store echo chamber size
            modularity_values.append(model.compute_modularity())  # ✅ Store modularity
            all_experiment_data.append((experiment_name, experiment_parameters, polarization_values.copy(), radicalization_values.copy(), average_degree_values.copy(), echo_chamber_values.copy(), modularity_values.copy()))  

            pol_fig = go.Figure()
            rad_fig = go.Figure()
            avg_degree_fig = go.Figure()
            echo_chamber_fig = go.Figure()
            modularity_fig = go.Figure()

            for i, (exp_name, params, pol_data, rad_data, average_degree_data, echo_chamber_data, modularity_values) in enumerate(all_experiment_data):
                pol_fig.add_trace(go.Scatter(y=pol_data, mode="lines+markers", name=f"Polarization ({exp_name})"))
                rad_fig.add_trace(go.Scatter(y=rad_data, mode="lines+markers", name=f"Radicalization ({exp_name})"))
                avg_degree_fig.add_trace(go.Scatter(y=average_degree_data, mode="lines+markers", name=f"Average Degree ({exp_name})"))
                echo_chamber_fig.add_trace(go.Scatter(y=echo_chamber_data, mode="lines+markers", name=f"Echo Chamber Extent ({exp_name})"))
                modularity_fig.add_trace(go.Scatter(y=modularity_values, mode="lines+markers", name=f"Modularity ({exp_name})"))

            pol_fig.update_layout(title="Polarization Over Time", xaxis_title="Rounds", yaxis_title="Polarization Level")
            rad_fig.update_layout(title="Radicalization Over Time", xaxis_title="Rounds", yaxis_title="Radicalization Level")
            avg_degree_fig.update_layout(title="Average Degree Over Time", xaxis_title="Rounds", yaxis_title="Average Degree")
            echo_chamber_fig.update_layout(title="Echo Chamber Extent Over Time", xaxis_title="Rounds", yaxis_title="Echo Chamber Extent")
            modularity_fig.update_layout(title="Modularity Over Time", xaxis_title="Rounds", yaxis_title="Modularity")

            processing = False
            return get_network_figure(model.graph), pol_fig, rad_fig, avg_degree_fig, echo_chamber_fig, modularity_fig, get_opinion_scatter_figure(), get_opinion_histogram(), f"Current Round: {current_round} / {new_num_rounds}"  

        polarization_values.append(model.measure_polarization())  # ✅ Store polarization
        radicalization_values.append(model.measure_radicalization())  # ✅ Store radicalization value
        average_degree_values.append(model.compute_average_degree())  # ✅ Store average degree
        echo_chamber_values.append(model.compute_echo_chamber_extent())  # ✅ Store echo chamber size
        modularity_values.append(model.compute_modularity())  # ✅ Store modularity

        # Propagation
        model.propagate_opinions()
        current_round += 1  

        if current_round != 0 and current_round % adjust_edges_every == 0:
            # if enable_attribute_vector is used, include them in similarity calculation;
            # else, just use opinions
            if "enabled" in enable_attribute_vector:
                model.adjust_edges_based_on_similarity(
                    threshold_connect=threshold_connect, threshold_disconnect=threshold_disconnect,
                    # use_random_recommendation=True,
                    sample_size_connect=5, sample_size_drop=dropout_size,
                    alpha_user=alpha_user, alpha_recommendation=alpha_recommendation,
                    use_content_based_recommendation=use_content_based, sample_size_content=num_content_based,
                    use_dissimilar_recommendation=use_dissimilar, sample_size_dissimilar=num_dissimilar,
                    use_collaborative_recommendation=use_collaborative, sample_size_collaborative=num_collaborative,
                    use_random_recommendation=use_random, sample_size_random=num_random,
                    
                    use_attribute_vector_recommendation=use_attribute_vector, sample_size_attribute_vector=num_attribute_vector,
                    use_hybrid_recommendation=use_hybrid, sample_size_hybrid=num_hybrid,
                    use_attribute_and_dissimilar_recommendation=use_attribute_dissimilar, sample_size_attribute_dissimilar=num_attribute_dissimilar    
                )
                print("Use attribute_vector for similarity calculations, alpha for similarity calculations:", alpha_user)
            else:
                model.adjust_edges_based_on_similarity(
                    threshold_connect=threshold_connect, threshold_disconnect=threshold_disconnect,
                    # use_random_recommendation=True,
                    sample_size_connect=5, sample_size_drop=dropout_size,
                    alpha_user=1.0, alpha_recommendation=1.0,
                    use_content_based_recommendation=use_content_based, sample_size_content=num_content_based,
                    use_dissimilar_recommendation=use_dissimilar, sample_size_dissimilar=num_dissimilar,
                    use_collaborative_recommendation=use_collaborative, sample_size_collaborative=num_collaborative,
                    use_random_recommendation=use_random, sample_size_random=num_random
                )
            print("Round: %s" % round)
            print("Thresholds: %s" % threshold_disconnect, threshold_connect)

            print(f"Recommendation Algorithms Used: ")
            print(f"🔹 Content-Based Recommendation: Used={use_content_based}, Nodes Recommended={num_content_based}")
            print(f"🔹 Dissimilar Recommendation: Used={use_dissimilar}, Nodes Recommended={num_dissimilar}")
            print(f"🔹 Collaborative Filtering: Used={use_collaborative}, Nodes Recommended={num_collaborative}")
            print(f"🔹 Random Recommendation: Used={use_random}, Nodes Recommended={num_random}")
            print(f"�� Attribute-Vector Recommendation: Used={use_attribute_vector}, Nodes Recommended={num_attribute_vector}")
            print(f"�� Hybrid Recommendation: Used={use_hybrid}, Nodes Recommended={num_hybrid}")
            print(f"�� Attribute-Dissimilar Recommendation: Used={use_attribute_dissimilar}, Nodes Recommended={num_attribute_dissimilar}")
            
            print("Weight for recommendation similarity: ", alpha_recommendation)
            print(f"Sample Dropout: num={dropout_size}")

        # prevent pics from being wiped out
        pol_fig = go.Figure(existing_pol_figure) if isinstance(existing_pol_figure, dict) else go.Figure()
        rad_fig = go.Figure(existing_rad_figure) if isinstance(existing_rad_figure, dict) else go.Figure()
        avg_degree_fig = go.Figure(existing_avg_degree_figure) if isinstance(existing_avg_degree_figure, dict) else go.Figure()
        echo_chamber_fig = go.Figure(existing_echo_chamber_figure) if isinstance(existing_echo_chamber_figure, dict) else go.Figure()
        modularity_fig = go.Figure(existing_modularity_figure) if isinstance(existing_modularity_figure, dict) else go.Figure()

        processing = False
        return get_network_figure(model.graph), pol_fig, rad_fig, avg_degree_fig, echo_chamber_fig, modularity_fig, get_opinion_scatter_figure(), get_opinion_histogram(), f"Current Round: {current_round} / {new_num_rounds}"  

    elif button_id == "finish-btn":
        print("🏁 Finishing simulation...")
        while current_round < new_num_rounds:
            polarization_values.append(model.measure_polarization())  # ✅ Store after propagation
            radicalization_values.append(model.measure_radicalization())  # ✅ Store radicalization value
            average_degree_values.append(model.compute_average_degree())  # ✅ Store average degree
            echo_chamber_values.append(model.compute_echo_chamber_extent())  # ✅ Store echo chamber size
            modularity_values.append(model.compute_modularity())  # ✅ Store modularity
            
            # Propagation
            model.propagate_opinions()
            current_round += 1  

            if current_round % adjust_edges_every == 0:
                # if enable_attribute_vector is used, include them in similarity calculation;
                # else, just use opinions
                if "enabled" in enable_attribute_vector:
                    model.adjust_edges_based_on_similarity(
                        threshold_connect=threshold_connect, threshold_disconnect=threshold_disconnect,
                        # use_random_recommendation=True,
                        sample_size_connect=5, sample_size_drop=dropout_size, 
                        alpha_user=alpha_user, alpha_recommendation=alpha_recommendation,
                        use_content_based_recommendation=use_content_based, sample_size_content=num_content_based,
                        use_dissimilar_recommendation=use_dissimilar, sample_size_dissimilar=num_dissimilar,
                        use_collaborative_recommendation=use_collaborative, sample_size_collaborative=num_collaborative,
                        use_random_recommendation=use_random, sample_size_random=num_random,
                        
                        use_attribute_vector_recommendation=use_attribute_vector, sample_size_attribute_vector=num_attribute_vector,
                        use_hybrid_recommendation=use_hybrid, sample_size_hybrid=num_hybrid,
                        use_attribute_and_dissimilar_recommendation=use_attribute_dissimilar, sample_size_attribute_dissimilar=num_attribute_dissimilar
                    )
                    print("Use attribute_vector for similarity calculations")
                else:
                    model.adjust_edges_based_on_similarity(
                        threshold_connect=threshold_connect, threshold_disconnect=threshold_disconnect,
                        # use_random_recommendation=True,
                        sample_size_connect=5, sample_size_drop=dropout_size, 
                        alpha_user=1.0, alpha_recommendation=1.0,
                        use_content_based_recommendation=use_content_based, sample_size_content=num_content_based,
                        use_dissimilar_recommendation=use_dissimilar, sample_size_dissimilar=num_dissimilar,
                        use_collaborative_recommendation=use_collaborative, sample_size_collaborative=num_collaborative,
                        use_random_recommendation=use_random, sample_size_random=num_random
                    )

                print("Finished testing")
                print("Thresholds: %s" % threshold_disconnect, threshold_connect)
            
                print(f"Recommendation Algorithms Used: ")
                print(f"🔹 Content-Based Recommendation: Used={use_content_based}, Nodes Recommended={num_content_based}")
                print(f"🔹 Dissimilar Recommendation: Used={use_dissimilar}, Nodes Recommended={num_dissimilar}")
                print(f"🔹 Collaborative Filtering: Used={use_collaborative}, Nodes Recommended={num_collaborative}")
                print(f"🔹 Random Recommendation: Used={use_random}, Nodes Recommended={num_random}")
                print(f"�� Attribute-Vector Recommendation: Used={use_attribute_vector}, Nodes Recommended={num_attribute_vector}")
                print(f"�� Hybrid Recommendation: Used={use_hybrid}, Nodes Recommended={num_hybrid}")
                print(f"�� Attribute-Dissimilar Recommendation: Used={use_attribute_dissimilar}, Nodes Recommended={num_attribute_dissimilar}")
                
                print("Weight for recommendation similarity: ", alpha_recommendation)
                print(f"Sample Dropout: num={dropout_size}")

        all_experiment_data.append((experiment_name, experiment_parameters, polarization_values.copy(), radicalization_values.copy(), average_degree_values.copy(), echo_chamber_values.copy(), modularity_values.copy()))  
        
        pol_fig = go.Figure()
        rad_fig = go.Figure()
        avg_degree_fig = go.Figure()
        echo_chamber_fig = go.Figure()
        modularity_fig = go.Figure()

        for i, (exp_name, params, pol_data, rad_data, avg_degree_data, echo_chamber_values, modularity_values) in enumerate(all_experiment_data):
            pol_fig.add_trace(go.Scatter(y=pol_data, mode="lines+markers", name=f"Polarization ({exp_name})"))
            rad_fig.add_trace(go.Scatter(y=rad_data, mode="lines+markers", name=f"Radicalization ({exp_name})"))
            avg_degree_fig.add_trace(go.Scatter(y=avg_degree_data, mode="lines+markers", name=f"Average Degree ({exp_name})"))
            echo_chamber_fig.add_trace(go.Scatter(y=echo_chamber_values, mode="lines+markers", name=f"Echo Chamber Extent ({exp_name})"))
            modularity_fig.add_trace(go.Scatter(y=modularity_values, mode="lines+markers", name=f"Modularity ({exp_name})"))

        pol_fig.update_layout(title="Polarization Over Time", xaxis_title="Rounds", yaxis_title="Polarization Level")
        rad_fig.update_layout(title="Radicalization Over Time", xaxis_title="Rounds", yaxis_title="Radicalization Level")
        avg_degree_fig.update_layout(title="Average Degree Over Time", xaxis_title="Rounds", yaxis_title="Average Degree")
        echo_chamber_fig.update_layout(title="Echo Chamber Extent Over Time", xaxis_title="Rounds", yaxis_title="Echo Chamber Extent")
        modularity_fig.update_layout(title="Modularity Over Time", xaxis_title="Rounds", yaxis_title="Modularity")

        processing = False
        return get_network_figure(model.graph), pol_fig, rad_fig, avg_degree_fig, echo_chamber_fig, modularity_fig, get_opinion_scatter_figure(), get_opinion_histogram(), f"Current Round: {current_round} / {new_num_rounds}"  
    
    elif button_id == "new-experiment-btn":
        print(f"🆕 Adding new experiment with N={new_n}, p={new_p}, num_rounds={new_num_rounds}")
        model = SocialNetworkModel(N=new_n, p=new_p, network_type="random", dynamic_edges=True, seed=seed)  
        
        current_round = 0  
        polarization_values = []
        radicalization_values = []
        average_degree_values = []
        echo_chamber_values = []
        modularity_values = []

        if "enabled" in enable_attribute_vector:
            print("Add attribute vectors, vector length is: ", attribute_vector_length)
            model.initialize_random_attributes(attribute_dim=attribute_vector_length)
        else:
            print("No attribute vectors.")

        # keep the data from the previous experiment
        pol_fig = go.Figure(existing_pol_figure) if isinstance(existing_pol_figure, dict) else go.Figure()
        rad_fig = go.Figure(existing_rad_figure) if isinstance(existing_rad_figure, dict) else go.Figure()
        avg_degree_fig = go.Figure(existing_avg_degree_figure) if isinstance(existing_avg_degree_figure, dict) else go.Figure()
        echo_chamber_fig = go.Figure(existing_echo_chamber_figure) if isinstance(existing_echo_chamber_figure, dict) else go.Figure()
        modularity_fig = go.Figure(existing_modularity_figure) if isinstance(existing_modularity_figure, dict) else go.Figure()

        processing = False
        return get_network_figure(model.graph), pol_fig, rad_fig, avg_degree_fig, echo_chamber_fig, modularity_fig, get_opinion_scatter_figure(), get_opinion_histogram(), f"Current Round: {current_round} / {new_num_rounds}" 

    processing = False
    return get_network_figure(model.graph), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), get_opinion_scatter_figure(), get_opinion_histogram(), f"Current Round: {current_round} / {new_num_rounds}"  


def export_experiment_data(data, output_dir="experiment_export"):
    os.makedirs(output_dir, exist_ok=True)

    if not data:
        print("No experiment data available to export.")
        return None, None
    
    print(f"📂 Exporting {len(data)} experiments to: {output_dir}")

    # Get experiment_name from the last experiment
    last_experiment_name = data[-1][0]

    # Sanitize folder name (optional: remove spaces, slashes, etc.)
    base_folder_name = last_experiment_name.replace(" ", "_")
    sub_dir = os.path.join(output_dir, base_folder_name)

    # Handle folder name duplication: add suffix _1, _2, ...
    count = 1
    while os.path.exists(sub_dir):
        sub_dir = os.path.join(output_dir, f"{base_folder_name}_{count}")
        count += 1

    os.makedirs(sub_dir)

    # === Prepare records
    records = []
    param_dict = {}

    for exp_name, params, pol, rad, deg, echo, mod in data:
        rounds = list(range(1, len(pol) + 1))
        for i in range(len(rounds)):
            records.append({
                "Experiment": exp_name,
                "Round": rounds[i],
                "Polarization": pol[i],
                "Radicalization": rad[i],
                "Average Degree": deg[i],
                "Echo Chamber": echo[i],
                "Modularity": mod[i]
            })
        param_dict[exp_name] = params

    # === Save Excel and JSON ===
    df = pd.DataFrame(records)
    excel_path = os.path.join(sub_dir, "experiment_metrics.xlsx")
    json_path = os.path.join(sub_dir, "experiment_parameters.json")

    try:
        # Write into Excel
        df.to_excel(excel_path, index=False)
        print(f"✅ Successfully wrote Excel file to {excel_path}")
    except Exception as e:
        print(f"❌ Failed to write Excel file: {e}")

    try:
        # Write into JSON
        with open(json_path, "w") as f:
            json.dump(param_dict, f, indent=4)
        print(f"✅ Successfully wrote JSON file to {json_path}")
    except Exception as e:
        print(f"❌ Failed to write JSON file: {e}")

    return excel_path, json_path


def get_opinion_scatter_figure():
    """ Generate scatter plot for opinion distribution. """
    reduced_opinions = model.get_reduced_opinions()  # get opinion distribution
    if reduced_opinions.shape[1] != 2:
        print("Warning: Reduced opinions do not have 2 components. Check PCA implementation.")
        return go.Figure()

    fig = go.Figure(data=[
        go.Scatter(
            x=reduced_opinions[:, 0], 
            y=reduced_opinions[:, 1],
            mode="markers",
            marker=dict(size=6, color="blue"),
            hoverinfo="text",
            text=[f"Node {i}" for i in range(len(reduced_opinions))]  # Hover 显示节点编号
        )
    ])
    
    fig.update_layout(
        title="Opinion Distribution (2D PCA Projection)",
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def get_opinion_histogram():
    """ Generate histogram for opinion distribution across all dimensions. """
    all_opinions = np.array([model.nodes[n].get_opinion() for n in model.nodes])  # (N, d)
    
    if all_opinions.shape[1] == 0:
        print("Warning: No opinion data available.")
        return go.Figure()

    fig = go.Figure()
    colors = plotly.colors.qualitative.Set1

    # draw histogram for each dimension
    for dim in range(all_opinions.shape[1]):
        fig.add_trace(go.Histogram(
            x=all_opinions[:, dim],
            name=f"Opinion Dim {dim+1}",
            opacity=0.5,
            marker=dict(color=colors[dim % len(colors)]),
            xbins=dict(
                start=-1.05,
                end=1.05,
                size=0.1
            )
        ))
        values, counts = np.unique(all_opinions[:, dim], return_counts=True)
        print(f"Dimension {dim+1}:")
        for val, count in zip(values, counts):
            print(f"  Value {val}: {count} times")

    fig.update_layout(
        title="Opinion Histogram (All Dimensions)",
        xaxis_title="Opinion Value",
        yaxis_title="Frequency",
        barmode="overlay",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_range=[-1, 1]
    )
    return fig

# === Run App ===
if __name__ == "__main__":
    threading.Timer(1.25, lambda: webbrowser.open_new("http://127.0.0.1:8050/") if not browser_opened else None).start()
    app.run(debug=True, use_reloader=False)