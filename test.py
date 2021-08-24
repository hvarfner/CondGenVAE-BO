tab_1_layout = html.Div([
            # html.H3('Patient Demographics'),

                html.Div([
                    html.Div([
                        html.H6('Gender', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='dem-graph-1',
                            figure={
                                'data': [

                                ],
                                'layout': {
                                    'title': 'Graph-1'
                                }
                            }
                        )
                    ], className="four columns"),

                    html.Div([
                        html.H6('Race', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='dem-graph-2',
                            figure={
                                'data': [

                                ],
                                'layout': {
                                    'title': 'Graph-2'
                                }
                            }
                        )
                    ], className="four columns"),

                    html.Div([
                        html.H6('Age', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='dem-graph-3',
                            figure={
                                'data': [

                                ],
                                'layout': {
                                    'title': 'Graph-3'
                                }
                            }
                        )
                    ], className="four columns")

                ], className="row", style={"margin": "1% 3%"}),

                html.Div([
                    html.Div([
                        html.H6('Race - Gender distribution', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='dem-graph-4',
                            figure={
                                'data': [

                                ],
                                'layout': {
                                    'title': 'Graph-4'
                                }
                            }
                        )
                        ], className="six columns"),

                    html.Div([
                        html.H6('Age - Gender Disrtibution', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='dem-graph-5',
                            figure={
                                'data': [

                                ],
                                'layout': {
                                    'title': 'Graph-5'
                                }
                            }
                        )
                    ], className="six columns"),
                ], className="row", style={"margin": "1% 3%"}),

                html.Div([
                    html.Div([
                        html.H6('Race - Age Disrtibution', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='dem-graph-6',
                            figure={
                                'data': [

                                ],
                                'layout': {
                                    'title': 'Graph-6'
                                }
                            }
                        )
                    ])
                ], className="row", style={"margin": "1% 3%"})

            ]
        )