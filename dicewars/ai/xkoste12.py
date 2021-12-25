import logging
import random

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.utils import possible_attacks
from dicewars.client.game.board import Board
from dicewars.client.game.area import Area

from copy import deepcopy


class AI:
    """Naive player agent

    This agent performs all possible moves in random order
    """

    def __init__(self, player_name, board, players_order, max_transfers):
        """
        Keyword arguments:
        player_name -- Player id (int)
        board -- Board, instance of dicewars.client.game.Board
        players_order -- Order of players (list of ints)
        max_transfers -- Max transfers per turn (int)
        """

        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.max_transfers = max_transfers
        self.players_order = players_order
        self.board = board

        self.MAXN_MAX_DEPTH = 3

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn (a single action)

        Keyword arguments:
        board -- Board, instance of dicewars.client.game.Board
        nb_moves_this_turn -- number of attacks made in this turn (int)
        nb_transfers_this_turn -- number of transfers made in this turn (int)
        nb_turns_this_game -- number of turns ended so far (int)
        time_left -- Time left in seconds (float)

        Returns:    
        BattleCommand() || EndTurnCommand() || TransferCommand()
        """ 

        self.board = board
        turn, src, tgt = self.get_best_move(board)

        if turn:
            return BattleCommand(src, tgt) if type(turn) is BattleCommand else TransferCommand(src, tgt)

        return EndTurnCommand()

    def get_best_move(self, board: Board):
        return self.maxn(board, 10)
    
    def maxn(self, board: Board, depth: int):
        moves = possible_attacks(board, self.player_name) + self.possible_transfers(board)

        if not moves or not depth: # No moves possible
            return self.evaluate_state(board), None

        for src, tgt in moves:
            new_board = deepcopy(board)
            self.simulate_move(new_board, src, tgt)

            evaluation, _ = self.maxn(board, depth - 1)

    def simulate_move(self, board: Board, src, tgt):
        pass

    def evaluate_state(self, board: Board):
        return random.sample(range(0, 10), 4)

    def possible_transfers(self, board: Board):
        for area in self.board.get_player_areas(self.player_name): 
            if not area.can_attack(): # can_attack means that area has more than one dice => can_attack <=> can_transfer
                continue

            neighbours = area.get_adjacent_areas_names()

            for adj in neighbours:
                adjacent_area = self.board.get_area(adj)
                if adjacent_area.get_owner_name() == self.player_name and adjacent_area.get_dice() != 8:
                    yield (area, adjacent_area)
