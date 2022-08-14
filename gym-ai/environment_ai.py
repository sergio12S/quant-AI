from typing import List, Tuple
import process_data
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy.random import RandomState
import vectorbt as vbt
from dataclasses import dataclass


@dataclass
class Positions:
    long: int = 0
    short: int = 1
    flat: int = 2

# TODO self.n_long, self.n_short, self.n_flat

class OhlcvEnv(gym.Env):

    def __init__(self, window_size: int = 50, show_trade: bool = True) -> None:
        self.positions = Positions()
        self.show_trade = show_trade
        self.fee = 0.0005
        self.seed()
        self.file_list = []
        # load_binance
        self.load_data_from_binance()
        # Download data from binance, vectortbt

        # n_features
        self.window_size = window_size
        self.n_features = self.df.shape[1]
        # TODO Why 4?
        self.shape = (self.window_size, self.n_features+4)

        # defines action space
        self.action_space = spaces.Discrete(len(self.positions.__dict__))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.shape,
            dtype=np.float32
        )
        self.done = False
        self.rand_episode = 0

    def load_data_from_binance(self) -> None:
        # Create object of Binance class
        data = vbt.BinanceData.download(
            ['BTCUSDT'],
            start='1000 minutes ago',
            end='now UTC',
            interval='5m'
        )
        # get data
        data = data.get(['Open', 'High', 'Low', 'Close', 'Volume'])
        data.columns = [column.lower() for column in data.columns]
        # create features objec of class FeatureExtractor
        extractor = process_data.FeatureExtractor(data)
        self.df = extractor.add_bar_features()
        # selected manual fetuares
        feature_list = [
            'bar_hc',
            'bar_ho',
            'bar_hl',
            'bar_cl',
            'bar_ol',
            'bar_co',
            'close']
        # Processing data
        self.df.dropna(inplace=True)  # drops NaN rows
        self.closingPrices = self.df['close'].values
        # redefine features with selected features
        self.df = self.df[feature_list].values

    def render_data_to_plot(self, mode: str = 'human', verbose: bool = False):
        # TODO what is mode? How to render?
        return None

    def seed(self, seed: int = 123) -> List[RandomState]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple:

        if self.done:
            return self.state, self.reward, self.done, {}
        self.reward = 0
        '''
        Description:

        action comes from the agent
        0 buy, 1 sell, 2 hold
        # single position can be opened per trade
        valid action sequence would be
        LONG : buy - hold - hold - sell
        SHORT : sell - hold - hold - buy
        invalid action sequence is just considered hold
        (e.g.) "buy - buy" would be considred "buy - hold"        
        '''

        if action == self.positions.long:  # buy
            if self.position == self.positions.flat:  # if previous position was flat
                self.position = self.positions.long  # update position to long
                self.entry_price = self.closingPrice  # maintain entry price
            elif self.position == self.positions.short:  # if previous position was short
                self.position = self.positions.flat  # update position to flat
                self.exit_price = self.closingPrice
                # calculate reward
                self.reward += ((self.entry_price - self.exit_price) /
                                self.exit_price + 1)*(1-self.fee)**2 - 1
                # evaluate cumulative return in krw-won
                self.krw_balance = self.krw_balance * (1.0 + self.reward)
                # clear entry price
                self.entry_price = 0
                # record number of short
                self.n_short += 1
        # vice versa for short trade
        elif action == 1:
            if self.position == self.positions.flat:
                self.position = self.positions.short
                self.entry_price = self.closingPrice
            elif self.position == self.positions.long:
                self.position = self.positions.flat
                self.exit_price = self.closingPrice
                self.reward += ((self.exit_price - self.entry_price) /
                                self.entry_price + 1)*(1-self.fee)**2 - 1
                self.krw_balance = self.krw_balance * (1.0 + self.reward)
                self.entry_price = 0
                self.n_long += 1

        # [coin + krw_won] total value evaluated in krw won
        if(self.position == self.positions.long):
            temp_reward = ((self.closingPrice - self.entry_price) /
                           self.entry_price + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        elif(self.position == self.positions.short):
            temp_reward = ((self.entry_price - self.closingPrice) /
                           self.closingPrice + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        else:
            temp_reward = 0
            new_portfolio = self.krw_balance

        self.portfolio = new_portfolio
        self.current_tick += 1
        if(self.show_trade and self.current_tick % 100 == 0):
            print(
                "Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
        self.history.append((action, self.current_tick,
                            self.closingPrice, self.portfolio, self.reward))
        self.updateState()
        if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
            self.done = True
            self.reward = self.get_profit()  # return reward at end of the game
        return self.state, self.reward, self.done, {'portfolio': np.array([self.portfolio]),
                                                    "history": self.history,
                                                    "n_trades": {'long': self.n_long, 'short': self.n_short}}

    def get_profit(self) -> float:
        if (self.position == self.positions.long):
            return ((self.closingPrice - self.entry_price) / self.entry_price + 1) * (1 - self.fee) ** 2 - 1
        elif self.position == self.positions.short:
            return ((self.entry_price - self.closingPrice) / self.closingPrice + 1) * (1 - self.fee) ** 2 - 1
        else:
            return 0.0

    def reset(self) -> np.ndarray:
        self.current_tick = 0
        print("start episode ... {0} at {1}" .format(
            self.rand_episode, self.current_tick))

        # positions
        self.n_long = 0
        self.n_short = 0

        # clear internal variables
        # keep buy, sell, hold action history
        self.history = []
        # initial balance, u can change it to whatever u like
        self.krw_balance = 100 * 10000
        self.portfolio = float(self.krw_balance)
        self.profit = 0

        self.position = self.positions.flat
        self.done = False

        self.updateState()
        return self.state

    def updateState(self) -> np.ndarray:
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]
        self.closingPrice = float(self.closingPrices[self.current_tick])
        prev_position = self.position
        one_hot_position = one_hot_encode(prev_position, 3)
        profit = self.get_profit()
        self.state = np.concatenate(
            (self.df[self.current_tick], one_hot_position, [profit]))
        return self.state
