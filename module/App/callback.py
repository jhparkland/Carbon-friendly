from dash import Dash, dcc, html, Output, Input, State, callback_context
from module.Firebase.firebase import FirebaseManager
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import os
import random
from .resource_info import resource_info
from dash.exceptions import PreventUpdate
from .layout import LayoutManager
from module.ElectricityMaps.electricity_maps import ElectricityMapsManager
from datetime import datetime, timedelta

class CallbackManager:
    """
    앱 콜백 스켈레톤 정의
    """
    def __init__(self, app, server):
        self.app = app # Dash에 대한 객체
        self.server = server # server에 대한 객체
        self.firebase = FirebaseManager() # Firebase에 대한 객체
        self.layout_manager = LayoutManager(self.app) #레이아웃 매니저 객체 생성
        self.pre_ev = 0 # ev 그래프 델타값 사용위해서
        self.cmp_ev = 0
        self.pre_emission = 0 # emission 그래프 델타값 사용위해서
        self.cmp_emission = 0
        self.pre_gfreq = 0 # gfreq 그래프 델타값 사용위해서
        self.cmp_gfreq = 0
        self.em = ElectricityMapsManager() # Electiricty 매니저 객체 생성
        self.graph_count = 0 # 초기호출 확인 변수
        self.elec_count = 0 # 초기호출 확인 변수
        self.resource_count = 0 # 초기호출 확인 변수
        self.resource_info_count = 0 #초기호출 확인 변수

        self.ev_all = 0 # 누적 전력사용량
        self.emission_all = 0 #누적 탄소배출량

    # RDB에서 컴퓨터 데이터를 읽어와서 정보를 반환하는 콜백함수
    def resources_callback(self):
        """
        컴퓨터 정보 콜백
        """
        @self.app.callback(
            Output('resources', 'children'),
            [Input('interval-component', 'n_intervals'),
             Input('zone', 'children')],
             prevent_initial_call=True
        )
        def update_resources_callback(n_intervals, children):
            ctx = callback_context
            if ctx.triggered_id == 'zone' and children is None and self.resource_count == 1:
                return self.layout_manager.resources
            else:
                self.resource_count = 1 # 초기호출 완료

                # RDB에서 컴퓨터 정보 읽어오기
                zone = self.firebase.read_data("main/zone")
                print(f'{zone}: 컴퓨터 정보 읽기 콜백')

                cpu_name = self.firebase.read_data(f"{zone}/CPU")
                cpu_use = self.firebase.read_data(f"{zone}/CPU Using")
                ram_size = self.firebase.read_data(f"{zone}/RAM")
                ram_use = self.firebase.read_data(f"{zone}/RAM Using")
                if zone == 'KR': country_name = '대한민국 서버'
                if zone == 'DE': country_name = '독일 서버'
                if zone == 'JP-TK': country_name = '일본 서버'
                if zone == 'FR': country_name = '프랑스서버'

                # 읽어온 정보로 직접 각 부분을 업데이트
                self.layout_manager.resources = dbc.Container(
                    dbc.Card([dbc.Col([
                        dbc.Row([html.H2("Computing Info 🖥️")], className="text-center"),
                        dbc.Row([html.Div(children=[f"{country_name}"],
                            style={
                                    'textAlign': 'center',
                                    'marginTop': '50px',
                                    'fontSize': '24px',  
                                    'fontWeight': 'bold',  
                            }
                        )]),
                        dbc.Row([html.Div(
                            [dcc.Graph(
                                id='cpu_architecture',
                                figure={
                                    'data': [
                                        go.Indicator(
                                                    mode='gauge+number',  
                                                    title={'text': f"CPU ARCHITECTURE: {cpu_name}", 'font':{'size':16}},  # Indicator 제목 설정
                                                    value=float(cpu_use[:-1]),
                                                    number = {'suffix': "%"},
                                                    gauge = {'axis': {'range': [0, 100]}}

                                            )
                                        ],  'layout': {
                                            'autosize': True,
                                            'margin': {'l': 40, 'r': 40, 't': 0, 'b': 0},  # 여백 설정
                                    } 
                                    }
                                ),])]),
                       
                        dbc.Row([html.Div([                
                            dcc.Graph(
                                        id='ram_size',
                                        figure={
                                            'data': [
                                                go.Indicator(
                                                    mode='gauge+number',  
                                                    title={'text':f"RAM SIZE: {ram_size}",'font':{'size':16}},  # Indicator 제목 설정
                                                    value=ram_use,
                                                    number = {'suffix': "%"},
                                                    gauge = {'axis': {'range': [0, 100]}}
                                        )
                                    ],  'layout': {
                                            'autosize': True,
                                            'margin': {'l': 40, 'r': 40, 't': 0, 'b': 0}  # 여백 설정
                         
                                    }    
                                        }
                                    ),
                        ])]),
                        ])
                    ]))
                # 업데이트된 정보를 tuple로 반환
                return self.layout_manager.resources
            

    def resource_info_callback(self):
        """
        pc 자원 정보 불러옴.
        """
        @self.app.callback(
            Output('resource_hid', 'children'),
            [Input('resource_info', 'n_intervals'),
             Input('zone', 'children')],
            prevent_initial_call=True
        )
        def update_resource(n_intervals,children):
            ctx = callback_context
            # 현재 하드웨어의 사용량(cpu, gpu, ram) 변수 선언 
            if ctx.triggered_id =='zone' and children is None and self.resource_info_count == 1:
                return None
            else:
                self.resource_info_count = 1 # 초기호출 완료

                # 현재 하드웨어의 사용량(cpu, gpu, ram) 변수 선언 
                resource = resource_info()
                curr_cpu_use = resource.cpuCurrUse()
                curr_ram_use =  resource.ramCurrUse()

                print(curr_cpu_use, curr_ram_use)

                # RDB에 현재 실행 중인 서버의 하드웨어 사용량 넣기
                zone = self.firebase.read_data("main/zone")

                print('컴퓨터 정보 쓰기 콜백')
                self.firebase.write_data(f"{zone}/CPU Using",curr_cpu_use)
                self.firebase.write_data(f"{zone}/RAM Using",curr_ram_use)

                return None
            
    # RDB에서 데이터를 읽어와서 그래프를 반환하는 콜백함수
    def graph_callback(self):
        """
        데이터, 그래프 콜백
        """
        @self.app.callback(
            [Output('ev', 'figure'),
            Output('emission', 'figure'),
            Output('gfreq', 'figure'),
            Output('sum_carbon_density', 'figure'),
            Output('sum_energy_output', 'figure')],
            [Input('interval-component', 'n_intervals'),  # 주기적으로 콜백을 트리거합니다
            Input('zone', 'children')],
            allow_duplicate=True
        )
        def update_graph_callback(n_intervals, children):
            ctx = callback_context
            if ctx.triggered_id == 'zone' and children is None and self.graph_count == 1:
                return self.layout_manager.ev_use_fig, self.layout_manager.carbon_emission_fig, self.layout_manager.gpu_freq_fig, self.layout_manager.carbon_emission_all_fig, self.layout_manager.ev_use_all_fig
            else:
                self.graph_count = 1 #초기호출 완료
                print('그래프 콜백')
                #현재 서버나라 읽어오기
                zone = self.firebase.read_data("main/zone")

                # Firebase에서 실시간 데이터 가져오기
                ev = self.firebase.read_data("optim/energyUsage") # 전력사용량
                cmp_ev = self.firebase.read_data(f"{zone}/cmpev")
                pre_ev = self.firebase.read_data(f"{zone}/preev")
                gfreq = self.firebase.read_data("optim/coreFreq") # GPU 주파수
                cmp_gfreq = self.firebase.read_data(f"{zone}/cmpgfreq")
                pre_gfreq = self.firebase.read_data(f"{zone}/pregfreq")
                carbon_data = self.em.carbon_intensity("carbon-intensity",zone = zone, format='latest') # 탄소배출량
                emission = carbon_data.get('carbonIntensity') * ev  # 탄소배출량 = 탄소밀집도 * 전력사용량
                cmp_emission = self.firebase.read_data(f"{zone}/cmpemission")
                pre_emission = self.firebase.read_data(f"{zone}/preemission")
                ev = ev * 1000 # kW -> W 단위변경
                self.emission_all = self.emission_all + emission # 누적 탄소배출량 더하기
                self.ev_all = self.ev_all + ev #누적 전력사용량 더하기 

                if(zone=='KR'): country ="대한민국"
                if(zone=='JP-TK'): country ="일본"
                if(zone=='DE'): country ="독일"
                if(zone=='FR'): country ="프랑스"

                # 델타값 비교 알고리즘
                if(cmp_ev != ev): 
                    pre_ev = cmp_ev
                    self.firebase.write_data(f"{zone}/preev", pre_ev)
                if(cmp_emission != emission):
                    pre_emission = cmp_emission
                    self.firebase.write_data(f"{zone}/preemission", pre_emission)
                if(cmp_gfreq != gfreq):
                    pre_gfreq = cmp_gfreq 
                    self.firebase.write_data(f"{zone}/pregfreq", pre_gfreq)
                self.firebase.write_data(f"{zone}/cmpev", ev)
                self.firebase.write_data(f"{zone}/cmpemission", emission)
                self.firebase.write_data(f"{zone}/cmpgfreq", gfreq)

                # 가져온 데이터를 레이아웃 데이터에 복사
                # 전력 사용량 그래프
                self.layout_manager.ev_use_fig = go.Figure(data = [go.Indicator(
                                                        mode="gauge+number+delta",
                                                        value=ev,
                                                        delta={'reference': pre_ev},
                                                        title={'text': "EV Usage(Wh)"},
                                                        number = {'suffix': "Wh"},
                                                        domain={'x': [0,1], 'y': [0,1]},
                                                        gauge={'axis': {'range': [0,5]}}
                )])
                self.layout_manager.ev_use_fig.update_layout(margin=dict(l=40, r=40, t=40, b=0), title=f'{country} 서버: 전력 사용량')

               # 누적 전력 사용량 그래프
                self.layout_manager.ev_use_all_fig = go.Figure(data = [go.Indicator(
                                                        mode="gauge+number+delta",
                                                        value=self.ev_all,
                                                        title={'text': "EV Usage(Wh)"},
                                                        delta={
                                                            'reference': self.ev_all - ev,
                                                            'relative': False,
                                                            'valueformat': '.2f',  # 소수점 두 자리까지 표시
                                                            'increasing': {'symbol': '&#9650;', 'color': 'green'},
                                                            'decreasing': {'symbol': '&#9660;', 'color': 'red'},
                                                        },
                                                        number = {'suffix': "Wh"},
                                                        domain={'x': [0,1], 'y': [0,1]},
                                                        gauge={'axis': {'range': [0,1000]}}
                )])
                self.layout_manager.ev_use_all_fig.update_layout(margin=dict(l=40, r=40, t=40, b=0), title='모든 서버: 누적 전력 사용량')


                #탄소 배출량 그래프
                self.layout_manager.carbon_emission_fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number+delta",
                    value=emission,
                    number = {'suffix': "g"},
                    gauge={
                        'shape':'bullet',
                        'axis':{'visible': True, 'range':[0,1]},
                    },
                    delta={'reference': pre_emission},
                    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
                )])
                # 타이틀을 그래프 위로 올리기
                self.layout_manager.carbon_emission_fig.update_layout(annotations=[dict(
                    text="Emission(g)",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.98,
                    align ="center",
                    font=dict(
                            size=20, # 원하는 크기로 조절
                        ),
                    )
                ])
                self.layout_manager.carbon_emission_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title=f'{country} 서버: 탄소 배출량')

                #누적 탄소배출량 그래프
                self.layout_manager.carbon_emission_all_fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number+delta",
                    value=self.emission_all,
                    number={'suffix': "g"},
                    delta={
                        'reference': self.emission_all - emission,
                        'relative': False,
                        'valueformat': '.3f',  # 소수점 두 자리까지 표시
                        'increasing': {'symbol': '&#9650;', 'color': 'green'},
                        'decreasing': {'symbol': '&#9660;', 'color': 'red'},
                    },
                    gauge={
                        'shape':'bullet',
                        'axis':{'visible': True, 'range':[0,500]},
                    },
                    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
                )])
                # 타이틀을 그래프 위로 올리기
                self.layout_manager.carbon_emission_all_fig.update_layout(annotations=[dict(
                    text="Emission(g)",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.98,
                    align ="center",
                    font=dict(
                            size=20, # 원하는 크기로 조절
                        ),
                    )
                ])
                self.layout_manager.carbon_emission_all_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title='모든 서버: 누적 탄소 배출량')

                # GPU 주파수 그래프
                self.layout_manager.gpu_freq_fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number+delta",
                    value=gfreq,
                    delta={'reference': pre_gfreq},
                    title={'text': "Frequency(Hz)"},
                    number = {'suffix': "Hz"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [0, 1530]}} # 실제 V100 GPU의 주파수 범위
                )])
                self.layout_manager.gpu_freq_fig.update_layout(margin=dict(l=40, r=40, t=40, b=0), title=f'{country} 서버: GPU 주파수')

                # 그래프 반환
                return self.layout_manager.ev_use_fig, self.layout_manager.carbon_emission_fig, self.layout_manager.gpu_freq_fig, self.layout_manager.carbon_emission_all_fig, self.layout_manager.ev_use_all_fig

    #일렉트리시티API 콜백
    def electricity_callback(self):
        @self.app.callback(
            Output('carbon_density', 'figure'),
            Output('energy_output', 'figure'),
            [Input('elec_interval-component', 'n_intervals'),  # 주기적으로 콜백을 트리거합니다
            Input('zone', 'children')],
            allow_duplicate=True
        )
        def update_electricity_callback(n_intervals, children):
            ctx = callback_context
            if ctx.triggered_id == 'zone' and children is None and self.elec_count == 1:
                return self.layout_manager.carbon_density_fig, self.layout_manager.energy_output_fig 
            else: 
                self.elec_count = 1 # 초기호출 완료
                print('elec 콜백')
                # 데이터 읽어오기
                zone = self.firebase.read_data("main/zone")
                carbon_data = self.em.carbon_intensity("carbon-intensity",zone = zone, format='latest')
                power_data_all = self.em.carbon_intensity("power-breakdown",zone = zone, format='latest')
                power_data = power_data_all.get("powerProductionBreakdown")
                if(zone=='KR'): country ="대한민국"
                if(zone=='JP-TK'): country ="일본"
                if(zone=='DE'): country ="독일"
                if(zone=='FR'): country ="프랑스"
                nuclear = power_data.get("nuclear") # 원자력
                geothermal = power_data.get("geothermal") # 지열
                biomass = power_data.get("biomass") # 바이오매스
                coal = power_data.get("coal") # 석탄
                wind = power_data.get("wind") # 바람
                solar = power_data.get("solar") # 태양
                hydro = power_data.get("hydro") # 수력
                hydro_discharge = power_data.get("hydro discharge") # 양수
                battery_discharge = power_data.get("battery discharge") # 배터리 용량
                gas = power_data.get("gas") # 가스
                oil = power_data.get("oil") # 오일
                unknown = power_data.get("unknown") # 알수없음
                # 시간읽어와서 형식바꾸기 ( +9시 )
                carbon_datetime = datetime.strptime(carbon_data.get("datetime"), "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(hours=9)
                power_datetime = datetime.strptime(power_data_all.get("datetime"), "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(hours=9)
                carbon_date = carbon_datetime.strftime("%Y-%m-%d")
                carbon_time = carbon_datetime.strftime("%H시")
                power_date = power_datetime.strftime("%Y-%m-%d")
                power_time = power_datetime.strftime("%H시")
                # 델타값 비교 알고리즘
                cmp_intensity = self.firebase.read_data(f"{zone}/cmpintensity")
                pre_intensity = self.firebase.read_data(f"{zone}/preintensity")
                if(cmp_intensity != carbon_data.get('carbonIntensity')): 
                    pre_intensity = carbon_data.get('carbonIntensity')
                    self.firebase.write_data(f"{zone}/preintensity", pre_intensity)
                    self.firebase.write_data(f"{zone}/cmpintensity", carbon_data.get('carbonIntensity'))

                #탄소밀집도 그래프
                self.layout_manager.carbon_density_fig = go.Figure(data=[go.Indicator(mode= "gauge+number+delta",
                                                                   title={'text': 'Carbon-Intensity(gCO2eq/kWh)'},
                                                                   number = {'suffix': "g"},
                                                                    value=carbon_data.get("carbonIntensity"),
                                                                    domain ={'x':[0,1], 'y': [0,1]},
                                                                    gauge={'axis': {'range': [0,1000]}},
                                                                    delta={'reference': pre_intensity},
                                                                    )]) 
                self.layout_manager.carbon_density_fig.update_layout(margin=dict(l=40, r=40, t=40, b=0), title=f'탄소밀집도: {country} ({carbon_date} {carbon_time})')

                # 에너지 출처 그래프
                self.layout_manager.energy_output_fig = go.Figure(data=go.Bar(
                    x = [nuclear, geothermal, biomass, coal, wind, solar, hydro, hydro_discharge, battery_discharge, gas, oil, unknown],
                    y = ['원자력', '지열', '바이오매스','석탄','바람','태양','수력','댐','배터리용량','가스','오일','알수없음'],
                    orientation='h'
                ))
                self.layout_manager.energy_output_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title=f'에너지 출처: {country} ({power_date} {power_time})')

                # 특정국가 평균 탄소 밀집도 < 현재 탄소 밀집도 조건 확인 Flag 전송 (임시기준)
                
                if(zone == 'KR'):
                    if(carbon_data.get('carbonIntensity') > 5000):
                        self.firebase.write_data("optim/request", True)
                    else: self.firebase.write_data("optim/request", False)
                if(zone == 'DE'):
                    if(carbon_data.get('carbonIntensity') > 420000):
                        self.firebase.write_data("optim/request", True)
                    else: self.firebase.write_data("optim/request", False)
                if(zone == 'JP-TK'):
                    if(carbon_data.get('carbonIntensity') > 480000):
                            self.firebase.write_data("optim/request", True)
                    else: self.firebase.write_data("optim/request", False)
                if(zone == 'FR'):
                    if(carbon_data.get('carbonIntensity') > 70000):
                        self.firebase.write_data("optim/request", True)
                    else: self.firebase.write_data("optim/request", False)
                
                return self.layout_manager.carbon_density_fig, self.layout_manager.energy_output_fig


    def electricity_callback2(self):
        @self.app.callback(
            Output('zone', 'children'),
            Input('elec_interval-component2', 'n_intervals')
        )
        def update_electricity_callback2(n_intervals):
            print('플래그체크 콜백')
            zone = None
            if self.firebase.read_data('optim/changed'):
                zone = self.firebase.read_data("main/zone")
                self.firebase.write_data('optim/changed', False)
                print(f'{zone}으로 바뀜')
            return zone
        
    # 지도콜백
    def geo_callback(self):
        @self.app.callback(
            Output('url', 'children'),
            Input('map', 'clickData'),  # 주기적으로 콜백을 트리거합니다
        )
        def update_url(clickData):
            pass
    
    # 스케줄 콜백
    def schedule_callback(self):
        """
        스케줄 콜백
        """
        @self.app.callback(
            Output('schedule', 'children'),
            Input('schedule_interval-component', 'n_intervals'),  # 주기적으로 콜백을 트리거합니다
        )
        def update_schedule_callback(n_intervals):
           pass