import logging
import random

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack
from dicewars.client.game.board import Board
from dicewars.client.game.area import Area

from copy import deepcopy
from typing import List, Tuple

import numpy as np


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
        src, tgt = self.get_best_move(board, nb_transfers_this_turn)

        if src and tgt:
            if board.get_area(tgt).get_owner_name() == self.player_name:
                return TransferCommand(src, tgt)
            else:
                return BattleCommand(src, tgt)

        return EndTurnCommand()

    def get_best_move(self, board: Board, transfers: int) -> Tuple[int, int]:
        return self.maxn(board, transfers, self.MAXN_MAX_DEPTH)[1]
    
    def maxn(self, board: Board, transfers: int, depth: int) -> Tuple[List[float], Tuple[int, int]]:
        moves = possible_attacks(board, self.player_name)

        if transfers < self.max_transfers: # Consider transfers only if we are still allowed to do them
            moves = moves + self.possible_transfers(board)

        # TODO: We have to check if the game is over or if we even have possible moves, I think

        if not moves or not depth: # We reached max depth or ran out of possible moves, return just the evaluation
            return self.evaluate_state(board), None

        evaluation = self.evaluate_state(board)

        for src, tgt in moves:
            if tgt.get_owner_name() != self.player_name:
                if probability_of_successful_attack(board, src, tgt) < 0.5:
                    continue

            new_board = deepcopy(board)
            new_transfers = transfers

            self.simulate_move(new_board, src.get_name(), tgt.get_name(), new_transfers)

            new_evaluation, _ = self.maxn(new_board, new_transfers, depth - 1)

            if new_evaluation[self.player_name - 1] > evaluation[self.player_name - 1]:
                best_src = src.get_name()
                best_tgt = tgt.get_name()

        return new_evaluation, (best_src, best_tgt)

    def simulate_move(self, board: Board, src_name: int, tgt_name: int, transfers: int) -> None:
        src = board.get_area(src_name)
        tgt = board.get_area(tgt_name)

        if tgt.get_owner_name() == self.player_name: # Simulating a transfer
            transfers = transfers + 1
            amount = tgt.get_dice() + src.get_dice() - 1

            if amount >= 8:
                tgt.set_dice(8)
                src.set_dice(amount - 7)
            else:
                tgt.set_dice(amount)
                src.set_dice(1)

        else: # Simulating an attack
            tgt.set_dice(src.get_dice() - 1)
            src.set_dice(1)
            tgt.set_owner(src.get_owner_name())

    def evaluate_state(self, board: Board) -> List[float]:
        return np.random.uniform(size=4)

    def possible_transfers(self, board: Board) -> Tuple[Area, Area]:
        for area in board.get_player_areas(self.player_name): 
            if not area.can_attack(): # can_attack means that area has more than one dice => can_attack <=> can_transfer
                continue

            neighbours = area.get_adjacent_areas_names()

            for adj in neighbours:
                adjacent_area = board.get_area(adj)
                if adjacent_area.get_owner_name() == self.player_name and adjacent_area.get_dice() != 8:
                    yield (area, adjacent_area)
