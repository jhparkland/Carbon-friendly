from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go

class LayoutManager:
    """
    앱 레이아웃 스켈레톤 정의
    """
    def __init__(self, app):
        self.app = app # Dash 인스턴스

        self.ev_use_fig = go.Figure(data = [go.Indicator(
                                                       mode="gauge+number",
                                                       title={'text': "EV Usage(Wh)"},
                                                       domain={'x': [0,1], 'y': [0,1]},
                                                       gauge={'axis': {'range': [0,1]}}
        )])
        self.ev_use_fig.update_layout(margin=dict(l=40, r=40, t=40, b=0), title=f'# 서버: 전력 사용량')

        self.ev_use_all_fig = go.Figure(data = [go.Indicator(
                                                       mode="gauge+number",
                                                       title={'text': "EV Usage(Wh)"},
                                                       domain={'x': [0,1], 'y': [0,1]},
                                                       gauge={'axis': {'range': [0,100]}}
        )])
        self.ev_use_all_fig.update_layout(margin=dict(l=40, r=40, t=40, b=0), title=f'모든 서버: 누적 전력 사용량')

        self.carbon_emission_fig = go.Figure(data=[go.Indicator( # 탄소배출량 그래프
                mode ="gauge+number",
                gauge={'shape':'bullet','axis':{'visible': False},},
                domain={'x': [0.1,1], 'y': [0.2,0.9]},

        )])
        # 타이틀을 그래프 위로 올리기
        self.carbon_emission_fig.update_layout(annotations=[dict(
                text="Emission(gCO2eq)",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.98,
                align ="center",
                font=dict(
                    size=20, # 원하는 크기로 조절
                )
            )
        ])
        self.carbon_emission_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title=f'# 서버: 탄소 배출량')

        self.carbon_emission_all_fig = go.Figure(data=[go.Indicator( # 누적 탄소배출량 그래프
                mode ="gauge+number",
                gauge={'shape':'bullet','axis':{'visible': False},},
                domain={'x': [0.1,1], 'y': [0.2,0.9]},

        )])
        # 타이틀을 그래프 위로 올리기
        self.carbon_emission_all_fig.update_layout(annotations=[dict(
                text="Emission(gCO2eq)",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.98,
                align ="center",
                font=dict(
                    size=20, # 원하는 크기로 조절
                )
            )
        ])
        self.carbon_emission_all_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title=f'모든 서버: 누적 탄소 배출량')

        self.gpu_freq_fig = go.Figure(data=[go.Indicator(mode = "gauge+number",
                                                        title = {'text': "Frequency(Hz)"},
                                                        domain = {'x': [0, 1], 'y': [0, 1]},
                                                        gauge={'axis': {'range': [0, 1530]}})]) # GPU 사용량 그래프
        self.gpu_freq_fig.update_layout(margin=dict(l=40, r=40, t=40, b=0), title=f'# 서버: GPU 주파수')

        
        self.carbon_density_fig = go.Figure(data=[go.Indicator(mode= "gauge+number",
                                                               title={'text': 'Carbon-Intensity(gCO2eq/kWh)'},
                                                               domain ={'x':[0,1], 'y': [0,1]},
                                                               gauge={'axis': {'range': [0,1000]}}
                                                               )]) # 탄소 밀도 그래프
        self.carbon_density_fig.update_layout(margin=dict(l=40, r=40, t=40, b=0), title=f'탄소 밀집도: 지역#')

        self.energy_output_fig = go.Figure(data=go.Bar(
                y = ['원자력', '지열', '바이오매스','석탄','바람','태양','수력','양수','배터리용량','가스','오일','알수없음'],
                orientation='h'
        ))
        self.energy_output_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title=f'에너지 출처: 지역#')

        self.geo = go.Figure(data=go.Choropleth(
            locations=['FRA', 'DEU', 'KOR', 'JPN'],  # 국가 코드
            z=[1, 2, 3, 4],  # 색상을 결정하는 값 (실제 데이터에 따라 변경 가능)
            text=['프랑스', '독일', '대한민국', '일본'],  # 각 국가 이름
            colorscale='Viridis',  # 색상 스케일
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            showscale=False  # 컬러바를 제거하기 위해 False로 설정
        ))

        self.geo.update_layout(
            width=3600,
            showlegend=False,  # 범례를 제거
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='equirectangular'
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        self.Logo = html.H1(
            "Carbon Watch",
            className="bg-dark text-white"
        )
        self.logo_path = "../../assets/img/logo.png" #로고 이미지 경로

        self.controls = dbc.Card(
            [
                dbc.CardBody(
                    [
                        self.Logo,  # 여기에 포함된 내용이 무엇인지에 따라 스타일 조정 필요
                        dbc.CardImg(src=self.logo_path, bottom=True, style={'width': '200px', 'height': '200px'})
                    ],
                    style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}  # Flexbox 스타일 적용
                )
            ]
        )

        self.resources = dbc.Container(
            dbc.Card([dbc.Col([
                dbc.Row([html.H2("Computing Info 🖥️")], className="text-center"),
                dbc.Row([html.Div("Server")]),
                dbc.Row([html.Div("CPU")]),
                dbc.Row([html.Div("GPU")]),
                dbc.Row([html.Div("RAM")])
            ])
        ])
        )
            
        # footer
        self.footer = html.Div([
                        html.P("© 2023 Data Science Lab All Rights Reserved."),
                        html.P("49315. 부산광역시 사하구 낙동대로 550번길 37(하단동)"),
                        html.P("동아대학교 공과대학1호관 4층 423호"),
                        html.P([    
                            html.A("Lab Website ", href="https://www.datasciencelabs.org/", target='_blank'),
                            html.A(" Contact Us ", href="https://github.com/datascience-labs", target='_blank'),
                            html.A(" GitHub ", href="https://github.com/jhparkland", target='_blank'),])
                        ],className="footer")


    def create_layout(self):
        """
        앱 레이아웃 생성

        Returns:
            _type_: 사전에 정의된 레이아웃 요소로 부터 레이아웃 생성
        """
        return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        self.controls,
                        dbc.Row(self.resources, id='resources')
                    ], width=3),

                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Card([html.Div(dcc.Graph(figure=self.carbon_emission_fig, id='emission'))], body=True, ), width=4),
                            dbc.Col(dbc.Card([html.Div(dcc.Graph(figure=self.ev_use_fig, id='ev'))], body=True, ), width=4),
                            dbc.Col(dbc.Card([html.Div(dcc.Graph(figure=self.gpu_freq_fig, id='gfreq'))], body=True, ), width=4),
                        ]),
                        dbc.Row([html.P()], style={'margin-top': '10px', 'margin-bottom': '10px'}),
                        dbc.Row([
                            dbc.Col(dbc.Card([html.Div(dcc.Graph(figure=self.carbon_emission_all_fig, id='sum_carbon_density'))], body=True, ), width=6),
                            dbc.Col(dbc.Card([html.Div(dcc.Graph(figure=self.ev_use_all_fig, id='sum_energy_output'))], body=True, ), width=6),
                        ]),
                        dbc.Row([
                            dbc.Col(dbc.Card([html.Div(dcc.Graph(figure=self.carbon_density_fig, id='carbon_density'))], body=True, ), width=6),
                            dbc.Col(dbc.Card([html.Div(dcc.Graph(figure=self.energy_output_fig, id='energy_output'))], body=True, ), width=6),
                        ]),
                        dbc.Row([
                            dbc.Col(dbc.Card([html.Div(dcc.Graph(figure=self.geo, id='map')), html.Div(id='url', style={'display':'none'})], body=True), width=12)
                        ]),
                        dbc.Row([html.P()], style={'margin-top': '10px', 'margin-bottom': '10px'}),
                    ], width=9),
                ]),
                self.footer,
                 # dcc.Interval을 추가하여 10초에 한 번씩 자동으로 콜백을 트리거합니다.
                dcc.Interval(
                            id='interval-component',
                            interval=5000,  # 5초마다 콜백을 트리거하도록 설정
                            n_intervals=0
                ),
                dcc.Interval(
                                id='elec_interval-component',
                                interval=1000 * 60 * 60,  # 1시간마다 콜백을 트리거하도록 설정
                                n_intervals=0
                ),
                dcc.Interval(
                                id='elec_interval-component2',
                                interval=3000,  # 3초마다 콜백을 트리거하도록 설정
                                n_intervals=0
                ),
                dcc.Interval(
                                id='resource_info',
                                interval=5000,  # 5초마다 콜백을 트리거하도록 설정
                                n_intervals=0
                ),   
                dcc.Interval(
                                id='schedule_interval-component',
                                interval=1000 * 60 * 5,  # 5분마다 콜백을 트리거하도록 설정
                                n_intervals=0
                ),            
                html.Div(id='schedule', style={'display':'none'}),
                html.Div(id='zone', style={'display':'none'}),
                html.Div(id='resource_hid', style={'display':'none'})
            ],fluid=True, className="dbc dbc-ag-grid",)