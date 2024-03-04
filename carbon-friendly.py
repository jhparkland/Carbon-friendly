from dash import Dash, dcc, html # Dash components
from module.App.callback import CallbackManager # callback functions
from module.App.layout import LayoutManager # layout functions
import dash_bootstrap_components as dbc # bootstrap components

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
    

app.title = 'Carbon friendly' # app title
app._favicon = 'assets/favicon/favicon.ico' # app favicon
server = app.server #  server connection


# create layout
layout = LayoutManager(app) 
app.layout = layout.create_layout()

# create callback
callback = CallbackManager(app, server)
callback.graph_callback() # 그래프 콜백
callback.resources_callback() # 컴퓨터자원 콜백
callback.resource_info_callback() # 컴퓨터자원 콜백
callback.electricity_callback() # 탄소밀집도,에너지출처 콜백
callback.geo_callback() # 지도 콜백
callback.electricity_callback2() # 플래그 체크 콜백

if __name__ == "__main__":
    # app.run(debug=True)
    app.run_server(debug=True, host='0.0.0.0', port=443)

