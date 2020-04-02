import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
from utg import UTG
import networkx as nx
import logging
from input_event import KeyEvent
from collections import deque
from PIL import Image
import cv2
cv2.ocl.setUseOpenCL(False)

POLICY_GREEDY_DFS = "dfs_greedy"
POLICY_GREEDY_BFS = "bfs_greedy"
POLICY_NONE = "none"

class DroidBotEnv(gym.Env):

    def __init__(self, droidbot):
        super(DroidBotEnv, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        metadata = {
            'render.modes': ['human'],
            'video.frames_per_second': None
        }
        self.seed()

        self.device = droidbot.device # use device to get current state for assigning actions to events
        self.input_manager = droidbot.input_manager # use input manager to send events to droidbot
        self.policy = droidbot.input_manager.policy

        # use policy to input events, get UTG
        self.possible_events = None # when we generate action_space, this is set
        self.policy_type = POLICY_GREEDY_DFS #set BFS/DFS/NONE for possible additional events
        self.add_unexplored = False # add unexplored actions to list of possible actions

        # Action size can change in be regenerated every step or set fixed
        #self.action_space
        self.action_space = spaces.Discrete(10)

        # Using image stack of past four states for observation space. See Humanoid paper for potential improvements
        self.stack_size = 4
        self.frames = deque([], maxlen=self.stack_size)
        self.height = 320
        self.width = 180
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, self.stack_size))

        # app stats in initial start state, subsequent resets need to stop and start the app
        self.reset_count = 0

    def step(self, action):
        import time
        start_time = time.time()
        info_dict = {}
        event = self.get_event(action)
        #print(time.time() - start_time)
        # returns None if index does not have corresponding event
        if event is not None:
            # do some checks prior to executing gym event
            new_event = self.policy.check_gym_event(event)
            self.input_manager.add_event(new_event)
            if event == new_event:
                info_dict['event_same'] = 1
        else:
            info_dict['event_same'] = 2
        #print(time.time() - start_time)

        state = self.device.get_current_state()
        #print(time.time() - start_time)

        state = self.handle_none_current_state(state)
        #print(time.time() - start_time)

        reward, done = self.get_reward_done(state)
        #print(time.time() - start_time)

        # get image for state and return it
        if done:
            img_stack = np.array(self.frames)
        else:
            state_img = self.get_image_state()
            self.frames.append(state_img)
            img_stack = np.array(self.frames)
            img_stack = np.moveaxis(img_stack, 0, -1)
       # print(time.time() - start_time)
        self.set_possible_events()

        #1/0
        return img_stack, reward, done, info_dict

    def get_event(self, action):
        if action >= len(self.possible_events):
            return None
        return self.possible_events[action]

    # if doing action size of variable number, use these functions
    # def step(self, action):
    #     info_dict = {}
    #     event = self.get_event(action)
    #     if event is not None:
    #         # do some checks prior to executing gym event
    #         new_event = self.policy.check_gym_event(event)
    #         self.input_manager.add_event(new_event)
    #         if event == new_event:
    #             info_dict['event_same'] = 1
    #     else:
    #         info_dict['event_same'] = 2
    #
    #     state = self.device.get_current_state()
    #     self.action_space
    #     reward, done = self.get_reward_done(state)
    #     # get image for state and return it
    #     state_img = self.get_image_state()
    #     self.frames.append(state_img)
    #     img_stack = np.array(self.frames)
    #     img_stack = np.moveaxis(img_stack, 0, -1)
    #
    #     return img_stack, reward, done, info_dict
    #
    # def get_event(self, action):
    #     return self.possible_events[action]
    #
    # # action space size shifts, generate after each step and restart
    # @property
    # def action_space(self):
    #     self.set_possible_events()
    #     return spaces.Discrete(len(self.possible_events))

    # update env with possible events
    def set_possible_events(self):
        state = self.device.get_current_state()
        events = state.get_possible_input()
        #print(events)
        self.possible_events = events
        self.events_probs = list(np.ones(len(events)) / len(events))
        # if humanoid, sort events by humanoid model
        if self.device.humanoid is not None:
            self.possible_events, self.events_probs = self.policy.sort_inputs_by_humanoid(self.possible_events)
        if self.add_unexplored:
            # get first unexplored event and insert it at beginning of events. is pushed to index 1 if using POLICY_GREED_BFS
            # if no unexplored event, None is inserted
            unexplored_event = self.get_unexplored_event()
            self.possible_events.insert(0, unexplored_event)
            self.events_probs.insert(0, 0)

        if self.policy_type == POLICY_GREEDY_BFS:
            self.possible_events.insert(0, KeyEvent(name="BACK"))
            self.events_probs.insert(0, 0)
        elif self.policy_type == POLICY_GREEDY_DFS:
            self.possible_events.append(KeyEvent(name="BACK"))
            self.events_probs.append(0)
        # print('FINAL SOURCE')
        # for event in self.possible_events:
        #     print(event.get_event_str(state))
        # print('END BASIC SOURCE')


    def reset(self):
        if self.reset_count > 0:
            #self.logger.info("Resetting env: calling stop ")
            event = self.policy.reset_stop()
            self.input_manager.add_event(event)
            event = self.policy.reset_home()
            self.input_manager.add_event(event)
            event = self.policy.reset_start()
            self.input_manager.add_event(event)
            #self.logger.info("Done resetting env ")

        self.reset_count += 1
        #state = self.device.get_current_state()
        state_img = self.get_image_state()
        for _ in range(self.stack_size):
            self.frames.append(state_img)
        img_stack = np.array(self.frames)
        img_stack = np.moveaxis(img_stack,0,-1)
        self.set_possible_events()

        # new state, thus generate new action space and events
        self.action_space
        return img_stack

    def get_image_state(self):
        img_path = self.device.take_screenshot()
        img = Image.open(img_path)
        img_array = np.array(img)
        #greyscale image
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        #resize image
        frame = cv2.resize(frame,(self.width,self.height), interpolation=cv2.INTER_AREA)
        return frame

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_reward_done(self, state):
        reward = 0.  # -0.01
        done = 0

        # if state is None:
        #     self.logger.warning("Failed to get current state in get_reward_done! Setting done = 1")
        #     reward = 0.
        #     done = 1
        # else:
        #     try:
        #         # State not visited before
        #         if not (self.policy.utg.is_state_reached(state)):
        #             reward += 0.04
        #         # Is a ending state?
        #         if len(state.get_possible_input()) == 0:
        #             # reward += -0.1
        #             done = 1
        #     except Exception as e:
        #         self.logger.warning("exception in get_reward_done: %s" % e)
        #         import traceback
        #         traceback.print_exc()
        #         reward = 0.
        #         done = 1

        return reward, done

    # return first unexplored event
    def get_unexplored_event_index(self):
        current_state = self.device.get_current_state()
        for i in range(len(self.possible_events)):
            if not self.policy.utg.is_event_explored(event=self.possible_events[i], state=current_state):
                #self.logger.info("Found an unexplored event, returning to agent")
                return i
        return None

    # return first unexplored event
    def get_unexplored_event(self):
        current_state = self.device.get_current_state()
        for input_event in self.possible_events:
            if not self.policy.utg.is_event_explored(event=input_event, state=current_state):
                #self.logger.info("Found an unexplored event, returning to agent")
                return input_event
        return None

    # return all unexplored events
    def get_unexplored_event_list(self):
        ret_list = []
        current_state = self.device.get_current_state()
        for input_event in self.possible_events:
            if not self.policy.utg.is_event_explored(event=input_event, state=current_state):
                ret_list.append(ret_list)
        return ret_list


    def handle_none_current_state(self, current_state):
        if current_state is None:
            self.logger.warning("Failed to get current state in handle_none! Sleep for 5 seconds then back event (per droidbot source code)")
            import time
            time.sleep(5)
            new_event = KeyEvent(name="BACK")
            self.input_manager.add_event(new_event)
            current_state = self.device.get_current_state()

        while current_state is None:
            self.logger.warning("Failed to get current state again! Resetting Env")
            self.reset()
            time.sleep(2)
            current_state = self.device.get_current_state()

        return current_state