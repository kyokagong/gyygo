from homepage.src.goAgent import TestGo
from homepage.src.tf_utils import Singleton

DEFAULT_AGENT = "default"


class AgentHandler(Singleton):

    def init_agent(self):
        self.agent_map = {}

        self.agent_map[DEFAULT_AGENT] = TestGo()


    def next_move(self, current_x, current_y):
        self.agent_map[DEFAULT_AGENT].user_move((current_x,current_y))
        return self.agent_map[DEFAULT_AGENT].agent_move()




