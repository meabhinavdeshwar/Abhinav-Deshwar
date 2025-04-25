#about.py

import dash_bootstrap_components as dbc
from dash import dcc, html

# -------------------------------------
# 🚀 Hero Section with Futuristic Airport Simulation Background
# -------------------------------------
hero_section = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🚀 SkyAI - The Future of Airport Intelligence", 
                        className="display-4 text-center animated fadeInDown",
                        style={"fontWeight": "bold", "color": "#FFD700"}),
                html.P(
                    "AI-powered analytics for smarter airport operations.",
                    className="lead text-center animated fadeIn",
                    style={"color": "#F8F9FA"}
                ),
                dbc.Button("🚀 Explore Dashboard", color="warning", href="/dashboard",
                           className="d-block mx-auto animated pulse infinite",
                           style={"fontSize": "18px", "borderRadius": "10px"})
            ], className="p-5 rounded",
                style={"background": "linear-gradient(to right, rgba(0,0,0,0.8), rgba(0,0,0,0.5))",
                       "backgroundImage": "url('/assets/futuristic_airport.jpg')", 
                       "backgroundSize": "cover", "backgroundPosition": "center", "borderRadius": "15px",
                       "boxShadow": "0px 4px 10px rgba(255, 215, 0, 0.4)"}
            )
        ], width=12),
    ])
], fluid=True)

# -------------------------------------
# 📌 About Section - Advanced Design with Key Highlights
# -------------------------------------
about_section = dbc.Container([
    hero_section,  

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("🔍 About SkyAI", className="bg-dark text-light"),
                dbc.CardBody([
                    html.H4("🌟 SkyAI: Revolutionizing Airports", className="card-title"),
                    html.Ul([
                        html.Li("💡 AI-driven predictive modeling for delays & efficiency."),
                        html.Li("🛫 Real-time flight simulation for better resource allocation."),
                        html.Li("📊 Data-driven insights for decision-makers."),
                    ], className="list-unstyled"),
                    html.P(
                        "SkyAI is an advanced AI-driven platform designed to optimize airport operations using real-time simulation, predictive analytics, and machine learning. "
                        "From delay forecasting to resource management, we transform airport efficiency through technology.",
                        className="card-text"
                    )
                ])
            ], className="mb-4 shadow-lg hover-glow rounded"),
            width=12
        )
    ], className="my-4"),
], fluid=True)

# -------------------------------------
# 🌍 Vision & Future Roadmap - Interactive & Concise
# -------------------------------------
roadmap_section = dbc.Container([
    html.Hr(),
    html.H2("🌍 Our Vision & Future Roadmap", className="text-center mb-4 animated fadeIn"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🚀 AI-Powered Automation", className="card-title"),
                    html.P("Leveraging AI to automate airport management and reduce inefficiencies.", className="card-text"),
                    html.Div([
                        html.I(className="fa fa-microchip fa-2x", style={"color": "#8E44AD", "marginBottom": "10px"}),
                        html.P("🔹 Smart gate scheduling & real-time analytics.", className="text-muted"),
                    ], className="text-center")
                ])
            ], className="text-center shadow-lg hover-glow rounded bg-light p-4")
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🛫 Predictive Delay Management", className="card-title"),
                    html.P("Using machine learning to enhance delay predictions and proactive solutions.", className="card-text"),
                    html.Div([
                        html.I(className="fa fa-clock fa-2x", style={"color": "#3498DB", "marginBottom": "10px"}),
                        html.P("🔹 Optimizing flight turnaround times with AI.", className="text-muted"),
                    ], className="text-center")
                ])
            ], className="text-center shadow-lg hover-glow rounded bg-light p-4")
        ], width=6),
    ], className="my-4"),
], fluid=True)

# -------------------------------------
# 🎭 Meet the Founders - Compact & Professional Design (Without Images)
# -------------------------------------
founders_data = [
    {
        "name": "Abhinav Deshwar",
        "role": "CTO | Backend & Frontend Developer",
        "description": "Architecting SkyAI’s AI-powered platform, optimizing backend systems, and ensuring seamless frontend experiences.",
        "linkedin": "http://www.linkedin.com/in/abhinavdeshwar"
    },
    {
        "name": "Om Raval",
        "role": "Data Analyst | Process Engineer",
        "description": "Enhancing SkyAI’s predictive models with data-driven insights, optimizing airport operations for efficiency.",
        "linkedin": "https://www.linkedin.com/in/om-raval"
    },
    {
        "name": "Aditya Deepak Thakkar",
        "role": "Project Manager",
        "description": "Leading execution and strategy, ensuring innovation aligns with industry needs for scalable airport intelligence.",
        "linkedin": "http://www.linkedin.com/in/adityathakkar032"
    },
    {
        "name": "Divyanshu Jaggi",
        "role": "Digital Creator | Data Analyst | Improvement Engineer",
        "description": "Driving process optimization through AI-powered insights to improve airport efficiency and decision-making.",
        "linkedin": "www.linkedin.com/in/divyanshu-jaggi"
    },
    {
        "name": "Yash Patel",
        "role": "Business Executor",
        "description": "Overseeing business strategy and execution, ensuring SkyAI meets industry demands and market expansion goals.",
        "linkedin": "http://www.linkedin.com/in/yashpatelyhp"
    },
    {
        "name": "Param Prakash Patel",
        "role": "Data Analyst | ML Specialist",
        "description": "Advancing machine learning models to refine delay predictions and optimize resource allocation.",
        "linkedin": "https://www.linkedin.com/in/param-patel1"
    }
]

founders_section = dbc.Container([
    html.Hr(),
    html.H2("🎭 Meet the Founders", className="text-center mb-4 animated fadeIn"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"👤 {founder['name']}", className="card-title text-center"),
                    html.P(f"🔹 {founder['role']}", className="card-text text-center", style={"fontSize": "14px", "fontWeight": "bold"}),
                    html.P(f"{founder['description']}", className="text-muted text-center", style={"fontSize": "12px"}),
                    dbc.Button("🔗 LinkedIn", color="primary", href=founder["linkedin"], target="_blank", className="d-block mx-auto btn-sm")
                ])
            ], className="shadow-lg hover-glow rounded mb-3")
        ], width=4) for founder in founders_data
    ], className="my-3", justify="center")
], fluid=True)

# -------------------------------------
# 📩 Contact & Social Media (Compact Design)
# -------------------------------------
footer_section = dbc.Container([
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H6("📧 Contact Us", className="text-center"),
            html.A("📩 skyai@ca.co.in", href="mailto:skyai@ca.co.in", 
                   className="d-block text-center", style={"color": "#007BFF", "fontSize": "14px"}),
        ], width=6),

        dbc.Col([
            html.H6("🔗 Follow Us", className="text-center"),
            html.A("🔗 LinkedIn", href="https://www.skyai.linkdin.in",
                   className="d-block text-center", style={"color": "#007BFF", "fontSize": "14px"}),
        ], width=6),
    ], className="my-2"),

    html.P("© 2024 SkyAI. All Rights Reserved.", className="text-center", style={"fontSize": "12px", "color": "#777"}),
], fluid=True, id="footer")

# -------------------------------------
# 🚀 Final Layout Assembly
# -------------------------------------
about_layout = dbc.Container([
    about_section,
    roadmap_section,
    founders_section,
    footer_section
], fluid=True)
