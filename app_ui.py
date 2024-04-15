from sympy import Symbol, Eq
import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Circle
from modulus.sym.domain.constraint import (
 PointwiseBoundaryConstraint,
 PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.utils.io import InferencerPlotter
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.pdes.linear_elasticity import LinearElasticityPlaneStress
from modulus.sym.eq.pdes.linear_elasticity import LinearElasticity
from modulus.sym.hydra.utils import compose
from modulus.sym.hydra import to_yaml
cfg = compose(config_path="conf", config_name="config")
# print(to_yaml(cfg))
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

import streamlit as st

# ---------- Start of Streamlit app

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: black;'> Panel </h1>",
            unsafe_allow_html=True)
st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

# --------- User input
st.markdown("<h3 style='text-align: center; color: black;'>USER INPUT </h3>",
            unsafe_allow_html=True
)




def run():
    # specify Panel properties
    E = 73.0 * 10**9  # Pa
    nu = 0.33
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))  # Pa
    mu_real = E / (2 * (1 + nu))  # Pa
    lambda_ = lambda_ / mu_real  # Dimensionless
    mu = 1.0  # Dimensionless

    # make list of nodes to unroll graph on
    le = LinearElasticityPlaneStress(lambda_=lambda_, mu=mu)
    elasticity_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("sigma_hoop")],
        output_keys=[
            Key("u"),
            Key("v"),
            Key("sigma_xx"),
            Key("sigma_yy"),
            Key("sigma_xy"),
        ],
        cfg=cfg.arch.fully_connected,
    )
    nodes = le.make_nodes() + [elasticity_net.make_node(name="elasticity_network")]

    # add constraints to solver
    # make geometry
    x, y, sigma_hoop = Symbol("x"), Symbol("y"), Symbol("sigma_hoop")
    panel_origin = (-0.5, -0.9)
    panel_dim = (1, 1.8)  # Panel width is the characteristic length.
    window_origin = (-0.125, -0.2)
    window_dim = (0.25, 0.4)
    panel_aux1_origin = (-0.075, -0.2)
    panel_aux1_dim = (0.15, 0.4)
    panel_aux2_origin = (-0.125, -0.15)
    panel_aux2_dim = (0.25, 0.3)
    hr_zone_origin = (-0.2, -0.4)
    hr_zone_dim = (0.4, 0.8)
    circle_nw_center = (-0.075, 0.15)
    circle_ne_center = (0.075, 0.15)
    circle_se_center = (0.075, -0.15)
    circle_sw_center = (-0.075, -0.15)
    circle_radius = 0.05
    panel = Rectangle(
        panel_origin, (panel_origin[0] + panel_dim[0], panel_origin[1] + panel_dim[1])
    )
    window = Rectangle(
        window_origin,
        (window_origin[0] + window_dim[0], window_origin[1] + window_dim[1]),
    )
    panel_aux1 = Rectangle(
        panel_aux1_origin,
        (
            panel_aux1_origin[0] + panel_aux1_dim[0],
            panel_aux1_origin[1] + panel_aux1_dim[1],
        ),
    )
    panel_aux2 = Rectangle(
        panel_aux2_origin,
        (
            panel_aux2_origin[0] + panel_aux2_dim[0],
            panel_aux2_origin[1] + panel_aux2_dim[1],
        ),
    )
    hr_zone = Rectangle(
        hr_zone_origin,
        (hr_zone_origin[0] + hr_zone_dim[0], hr_zone_origin[1] + hr_zone_dim[1]),
    )
    circle_nw = Circle(circle_nw_center, circle_radius)
    circle_ne = Circle(circle_ne_center, circle_radius)
    circle_se = Circle(circle_se_center, circle_radius)
    circle_sw = Circle(circle_sw_center, circle_radius)
    corners = (
        window - panel_aux1 - panel_aux2 - circle_nw - circle_ne - circle_se - circle_sw
    )
    window = window - corners
    geo = panel - window
    hr_geo = geo & hr_zone

    # Parameterization
    characteristic_length = panel_dim[0]
    characteristic_disp = 0.001 * window_dim[0]
    sigma_normalization = characteristic_length / (mu_real * characteristic_disp)
    sigma_hoop_lower = 46 * 10**6 * sigma_normalization
    sigma_hoop_upper = 56.5 * 10**6 * sigma_normalization
    sigma_hoop_range = (sigma_hoop_lower, sigma_hoop_upper)
    param_ranges = {sigma_hoop: sigma_hoop_range}
    inference_param_ranges = {sigma_hoop: 46 * 10**6 * sigma_normalization}

    # bounds
    bounds_x = (panel_origin[0], panel_origin[0] + panel_dim[0])
    bounds_y = (panel_origin[1], panel_origin[1] + panel_dim[1])
    hr_bounds_x = (hr_zone_origin[0], hr_zone_origin[0] + hr_zone_dim[0])
    hr_bounds_y = (hr_zone_origin[1], hr_zone_origin[1] + hr_zone_dim[1])

    # make domain
    domain = Domain()

    # add inferencer data
    invar_numpy = geo.sample_interior(
        100,
        bounds={x: bounds_x, y: bounds_y},
        parameterization=inference_param_ranges,
    )
    point_cloud_inference = PointwiseInferencer(
        nodes=nodes,
        invar=invar_numpy,
        output_names=["u", "v", "sigma_xx", "sigma_yy", "sigma_xy"],
        batch_size=4096,
        plotter = InferencerPlotter()
    )
    domain.add_inferencer(point_cloud_inference, "inf_data")
    invar, outpred = point_cloud_inference.eval_epoch()
    input_sample = {'x': invar['x'], 'y': invar['y']}
    return point_cloud_inference.plotter(input_sample, outpred)
   

    

col1, col2 = st.columns([1,1])


val = col1.number_input(
    "Value 1",
    value=None,
    min_value=0,
    max_value=100,
    placeholder="Enter Value ...", 
    format="%d",
    label_visibility="visible",
)

val1 = col1.number_input(
    "Value 2",
    value=None,
    min_value=0,
    max_value=100,
    placeholder="Enter Value ...", 
    format="%d",
    label_visibility="visible",
)

submit = st.button("Predict", help="Click here to start prediction", 
                    type="primary", use_container_width=True,
)
st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

# ------- Prediction section 
st.markdown(
    "<h1 style='text-align: center; color: black;'>PREDICTION </h1>",
    unsafe_allow_html=True
)


if submit == True: 
    # Create a dictionary with the user's input values  
    user_input = {  
        "val": val,
        "Val1": val1, 
    }
    result = run()
    for x in result:
        img, name = x
        buffer = BytesIO()
        img.savefig(buffer, format='png')
        buffer.seek(0)
        image = Image.open(buffer)
        st.image(image)
        
#     print(type(result), len(result))

 