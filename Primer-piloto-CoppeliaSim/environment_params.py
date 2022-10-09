class ParametersEnvironments():
    def __init__(self, environment):
        self.agent_model = [""]
        self.list_allowed_forward_actions = []
        self.dict_posible_outcomes = {}
        self.agent_index = 0

        if (environment == "Train_4x4"):
            self.define_train_4x4()
        elif (environment == "Train_6x6"):
            self.define_train_6x6()
        elif (environment == "Test_4x4"):
            self.define_test_4x4()
        elif (environment == "Test_6x6"):
            self.define_test_6x6()

    def define_train_4x4(self):
        self.agent_model = ["/Robot-4.ttm"]
        self.list_allowed_forward_actions = [
            (6.7, 6.7, 90),
            (6.7, 5.2, 0),
            (5.2, 5.2, 270),
            (5.2, 6.7, 0),
            (3.8, 6.7, 0),
            (2.4, 6.7, 90),
            (2.4, 5.2, 90),
            (2.4, 3.8, 180),
            (3.8, 3.8, 90),
            (3.8, 2.4, 0),
            (5.2, 5.2, 90),
            (5.2, 3.8, 180),
            (6.7, 3.8, 90),
            (6.7, 2.4, 0),
            (5.2, 2.4, 0),
            (2.4, 2.4, 0),
            (3.8, 2.4, 180),
            (5.2, 2.4, 180),
            (6.7, 2.4, 270),
            (6.7, 3.8, 0),
            (5.2, 3.8, 270),
            (5.2, 5.2, 180),
            (6.7, 5.2, 270),
            (3.8, 2.4, 270),
            (3.8, 3.8, 0),
            (3.8, 3.8, 270),
            (2.4, 3.8, 270),
            (2.4, 5.2, 270),
            (2.4, 6.7, 180),
            (3.8, 6.7, 180),
            (5.2, 6.7, 90),
            (3.8, 5.2, 90)
        ]
        self.dict_posible_outcomes = {
            (6.7, 6.7, 0): 'a;',
            (6.7, 6.7, 90): 'w;',
            (6.7, 5.2, 90): 'd;',
            (6.7, 5.2, 0): 'w;',
            (5.2, 5.2, 0): 'a;d',
            (5.2, 5.2, 270): 'w;',
            (5.2, 6.7, 270): 'a;',
            (5.2, 6.7, 0): 'w;',
            (3.8, 6.7, 0): 'w;',
            (2.4, 6.7, 0): 'a;',
            (2.4, 6.7, 90): 'w;',
            (2.4, 5.2, 90): 'w;',
            (2.4, 3.8, 90): 'a;',
            (2.4, 3.8, 180): 'w;',
            (3.8, 3.8, 180): 'd;',
            (3.8, 3.8, 90): 'w;',
            (3.8, 2.4, 90): 'd;',
            (3.8, 2.4, 0): 'w;',
            (5.2, 5.2, 90): 'w;',
            (5.2, 3.8, 90): 'a;',
            (5.2, 3.8, 180): 'w;',
            (6.7, 3.8, 180): 'd;',
            (6.7, 3.8, 90): 'w;',
            (6.7, 2.4, 90): 'd;',
            (6.7, 2.4, 0): 'w;',
            (5.2, 2.4, 0): 'w;',
            (2.4, 2.4, 0): 'w;'
        }

    def define_test_4x4(self):
        self.agent_model = [
            "/Robot-1.ttm",
            "/Robot-2.ttm",
            "/Robot-3.ttm",
            "/Robot-4.ttm",
            "/Robot-5.ttm",
            "/Robot-6.ttm",
            "/Robot-7.ttm",
            "/Robot-8.ttm",
            "/Robot-9.ttm",
            "/Robot-10.ttm",
            "/Robot-11.ttm",
            "/Robot-12.ttm"
        ]
        self.list_allowed_forward_actions = [
            (6.7, 6.7, 90),
            (6.7, 5.2, 0),
            (5.2, 5.2, 270),
            (5.2, 6.7, 0),
            (3.8, 6.7, 0),
            (2.4, 6.7, 90),
            (2.4, 5.2, 90),
            (2.4, 3.8, 180),
            (3.8, 3.8, 90),
            (3.8, 2.4, 0),
            (5.2, 5.2, 90),
            (5.2, 3.8, 180),
            (6.7, 3.8, 90),
            (6.7, 2.4, 0),
            (5.2, 2.4, 0),
            (2.4, 2.4, 0),
            (3.8, 2.4, 180),
            (5.2, 2.4, 180),
            (6.7, 2.4, 270),
            (6.7, 3.8, 0),
            (5.2, 3.8, 270),
            (5.2, 5.2, 180),
            (6.7, 5.2, 270),
            (3.8, 2.4, 270),
            (3.8, 3.8, 0),
            (3.8, 3.8, 270),
            (2.4, 3.8, 270),
            (2.4, 5.2, 270),
            (2.4, 6.7, 180),
            (3.8, 6.7, 180),
            (5.2, 6.7, 90),
            (3.8, 5.2, 90)
        ]

    def define_train_6x6(self):
        self.close2objective_list = [
            (0, 0, 0),
            (0, 1.5, 0.05),
            (0, 3, 0.1),
            (0, 4.5, 0.15),
            (0, 6, 0.05),
            (0, 7.5, 0),
            (1.5, 0, 0.05),
            (1.5, 1.5, 0.1),
            (1.5, 3, 0.15),
            (1.5, 4.5, 0.2),
            (1.5, 6, 0.1),
            (1.5, 7.5, 0.05),
            (3, 0, 0),
            (3, 1.5, 0.15),
            (3, 3, 0.2),
            (3, 4.5, 0.3),
            (3, 6, 0.2),
            (3, 7.5, 0.1),
            (4.5, 0, 0.05),
            (4.5, 1.5, 0.1),
            (4.5, 3, 0.2),
            (4.5, 4.5, 0.25),
            (4.5, 6, 0.15),
            (4.5, 7.5, 0.05),
            (6, 0, 0.05),
            (6, 1.5, 0.1),
            (6, 3, 0.15),
            (6, 4.5, 0.2),
            (6, 6, 0.15),
            (6, 7.5, 0.1),
            (7.5, 0, 0),
            (7.5, 1.5, 0.05),
            (7.5, 3, 0.1),
            (7.5, 4.5, 0.1),
            (7.5, 6, 0.05),
            (7.5, 7.5, 0)
        ]
        self.positions_list = [
            0,
            1.5,
            3,
            4.5,
            6,
            7.5
        ]
        self.dict_posible_outcomes = {
            (0, 0, 90): 'd;',
            (0, 0, 0): 'w;',
            (1.5, 0, 0): 'a;',
            (1.5, 0, 90): 'w;',
            (1.5, 1.5, 90): 'd;',
            (1.5, 1.5, 0): 'w;',
            (3, 1.5, 0): 'a;',
            (3, 1.5, 90): 'w;',
            (3, 3, 90): 'w;',
            (0, 7.5, 0): 'd;',
            (0, 7.5, 270): 'w;',
            (0, 6, 270): 'a;',
            (0, 6, 0): 'w;',
            (1.5, 6, 0): 'w;',
            (3, 6, 0): 'd;',
            (3, 6, 270): 'w;',
            (7.5, 0, 90): 'a;',
            (7.5, 0, 180): 'w;',
            (6, 0, 180): 'd;',
            (6, 0, 90): 'w;',
            (6, 1.5, 90): 'w;',
            (6, 3, 90): 'a;',
            (6, 3, 180): 'w;',
            (4.5, 3, 180): 'w;',
            (3, 3, 180): 'd;',
            (7.5, 7.5, 180): 'w;',
            (6, 7.5, 180): 'a;',
            (6, 7.5, 270): 'w;',
            (6, 6, 270): 'w;',
            (6, 4.5, 270): 'd;',
            (6, 4.5, 180): 'w;',
            (4.5, 4.5, 180): 'w;'
        }
        self.list_allowed_forward_actions = [
            (0, 0, 0),
            (0, 1.5, 0),
            (0, 1.5, 90),
            (0, 3, 0),
            (0, 3, 90),
            (0, 3, 270),
            (0, 4.5, 0),
            (0, 4.5, 270),
            (0, 6, 90),
            (0, 7.5, 270),
            (0, 6, 0),
            (1.5, 0, 180),
            (1.5, 0, 90),
            (1.5, 1.5, 180),
            (1.5, 1.5, 270),
            (1.5, 1.5, 0),
            (1.5, 3, 180),
            (1.5, 3, 90),
            (1.5, 4.5, 180),
            (1.5, 4.5, 270),
            (1.5, 4.5, 0),
            (1.5, 6, 0),
            (1.5, 6, 90),
            (1.5, 6, 180),
            (1.5, 7.5, 0),
            (1.5, 7.5, 270),
            (3, 0, 0),
            (3, 1.5, 180),
            (3, 1.5, 90),
            (3, 3, 0),
            (3, 3, 270),
            (3, 3, 90),
            (3, 6, 90),
            (3, 6, 180),
            (3, 6, 270),
            (3, 7.5, 0),
            (3, 7.5, 180),
            (3, 7.5, 270),
            (4.5, 0, 180),
            (4.5, 0, 90),
            (4.5, 1.5, 0),
            (4.5, 1.5, 90),
            (4.5, 1.5, 180),
            (4.5, 3, 0),
            (4.5, 3, 90),
            (4.5, 3, 270),
            (4.5, 3, 180),
            (4.5, 4.5, 0),
            (4.5, 4.5, 90),
            (4.5, 4.5, 270),
            (4.5, 4.5, 180),
            (4.5, 6, 0),
            (4.5, 6, 90),
            (4.5, 6, 270),
            (4.5, 7.5, 270),
            (4.5, 7.5, 180),
            (6, 0, 0),
            (6, 0, 90),
            (6, 1.5, 90),
            (6, 1.5, 0),
            (6, 1.5, 90),
            (6, 1.5, 270),
            (6, 3, 180),
            (6, 3, 270),
            (6, 3, 0),
            (6, 4.5, 90),
            (6, 4.5, 180),
            (6, 6, 180),
            (6, 6, 90),
            (6, 6, 270),
            (6, 7.5, 0),
            (6, 7.5, 270),
            (7.5, 0, 180),
            (7.5, 1.5, 180),
            (7.5, 1.5, 90),
            (7.5, 3, 90),
            (7.5, 3, 180),
            (7.5, 3, 270),
            (7.5, 4.5, 270),
            (7.5, 4.5, 90),
            (7.5, 6, 270),
            (7.5, 6, 90),
            (7.5, 7.5, 270),
            (7.5, 7.5, 180)
        ]
        self.agent_model = [
            "/Robot-1.ttm",
            "/Robot-2.ttm",
            "/Robot-3.ttm",
            "/Robot-4.ttm"
        ]
        self.goal_object_position_list = [
            (3, 4.5)
        ]

    def define_test_6x6(self):
        self.agent_model = [
            "/Robot-1.ttm",
            "/Robot-2.ttm",
            "/Robot-3.ttm",
            "/Robot-4.ttm",
            "/Robot-5.ttm",
            "/Robot-6.ttm",
            "/Robot-7.ttm",
            "/Robot-8.ttm",
            "/Robot-9.ttm",
            "/Robot-10.ttm"
        ]        
        self.list_allowed_forward_actions = [
            (0, 0, 0),
            (0, 1.5, 0),
            (0, 1.5, 90),
            (0, 3, 0),
            (0, 3, 90),
            (0, 3, 270),
            (0, 4.5, 0),
            (0, 4.5, 270),
            (0, 6, 90),
            (0, 7.5, 270),
            (0, 6, 0),
            (1.5, 0, 180),
            (1.5, 0, 90),
            (1.5, 1.5, 180),
            (1.5, 1.5, 270),
            (1.5, 1.5, 0),
            (1.5, 3, 180),
            (1.5, 3, 90),
            (1.5, 4.5, 180),
            (1.5, 4.5, 270),
            (1.5, 4.5, 0),
            (1.5, 6, 0),
            (1.5, 6, 90),
            (1.5, 6, 180),
            (1.5, 7.5, 0),
            (1.5, 7.5, 270),
            (3, 0, 0),
            (3, 1.5, 180),
            (3, 1.5, 90),
            (3, 3, 0),
            (3, 3, 270),
            (3, 3, 90),
            (3, 6, 90),
            (3, 6, 180),
            (3, 6, 270),
            (3, 7.5, 0),
            (3, 7.5, 180),
            (3, 7.5, 270),
            (4.5, 0, 180),
            (4.5, 0, 90),
            (4.5, 1.5, 0),
            (4.5, 1.5, 90),
            (4.5, 1.5, 180),
            (4.5, 3, 0),
            (4.5, 3, 90),
            (4.5, 3, 270),
            (4.5, 3, 180),
            (4.5, 4.5, 0),
            (4.5, 4.5, 90),
            (4.5, 4.5, 270),
            (4.5, 4.5, 180),
            (4.5, 6, 0),
            (4.5, 6, 90),
            (4.5, 6, 270),
            (4.5, 7.5, 270),
            (4.5, 7.5, 180),
            (6, 0, 0),
            (6, 0, 90),
            (6, 1.5, 90),
            (6, 1.5, 0),
            (6, 1.5, 90),
            (6, 1.5, 270),
            (6, 3, 180),
            (6, 3, 270),
            (6, 3, 0),
            (6, 4.5, 90),
            (6, 4.5, 180),
            (6, 6, 180),
            (6, 6, 90),
            (6, 6, 270),
            (6, 7.5, 0),
            (6, 7.5, 270),
            (7.5, 0, 180),
            (7.5, 1.5, 180),
            (7.5, 1.5, 90),
            (7.5, 3, 90),
            (7.5, 3, 180),
            (7.5, 3, 270),
            (7.5, 4.5, 270),
            (7.5, 4.5, 90),
            (7.5, 6, 270),
            (7.5, 6, 90),
            (7.5, 7.5, 270),
            (7.5, 7.5, 180)
        ]
        self.goal_object_position_list = [
            (3, 4.5)
        ]        
        self.positions_list = [
            0,
            1.5,
            3,
            4.5,
            6,
            7.5
        ]       

    def update_agent_index(self):
        if self.agent_index < len(self.agent_model):
            self.agent_index += 1

    def get_agent_model(self):
        self.update_agent_index()
        return self.agent_model[self.agent_index - 1]
