import chess
import torch
import random
from typing import Optional
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from chess_tournament import Player

class TransformerPlayer(Player):
    def __init__(self, name: str, model_path: str = "belpekkan/chess_T5_seq2seq"):
        #v4
        # TransformerPlayer("Student") works without any extra arguments
        super().__init__(name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # keep track of positions we've already been in to avoid repeating
        self._seen_positions: set = set()

    # OVERRIDE 1 - CHECKS FOR 2 MOVE CHECKMATES
    def _find_checkmate(self, board: chess.Board) -> Optional[chess.Move]:
        # check if model can checkmate in 1
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()

        # check if model can force checkmate in 2
        for move in board.legal_moves:
            board.push(move)

            # skip stalemate
            if board.is_stalemate():
                board.pop()
                continue

            opponent_moves = list(board.legal_moves)

            if not opponent_moves:
                board.pop()
                continue

            all_replies_mated = True
            for opp_move in opponent_moves:
                board.push(opp_move)

                # check whether model can have a mate-in-1 from this position
                we_have_mate = False
                for our_reply in board.legal_moves:
                    board.push(our_reply)
                    if board.is_checkmate():
                        board.pop()
                        we_have_mate = True
                        break
                    board.pop()

                board.pop()  # undo opponent move

                if not we_have_mate:
                    all_replies_mated = False
                    break  # no need to check remaining opponent replies

            board.pop()  # undo our candidate move

            if all_replies_mated:
                return move

        return None

    # OVERRIDE 2 — PROMOTION
    
    def _find_promotion(self, board: chess.Board) -> Optional[chess.Move]:
        # collect all queen promotions
        promotions = [
            m for m in board.legal_moves
            if m.promotion == chess.QUEEN
        ]
        if not promotions:
            return None
        # prefer a promotion that also gives check
        for move in promotions:
            board.push(move)
            gives_check = board.is_check()
            board.pop()
            if gives_check:
                return move
        return promotions[0]

    # OVERRIDE 3 — WINNING POSITIONS: prefer captures and checks
    
    def _material_balance(self, board: chess.Board) -> int:
        # rough centipawn count from our perspective
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }
        score = 0
        for piece_type, value in self.PIECE_VALUES.items():
            score += value * len(board.pieces(piece_type, board.turn))
            score -= value * len(board.pieces(piece_type, not board.turn))
        return score

    def _filter_winning_moves(self, board: chess.Board, moves: list) -> list:
        # if we're up by about a rook or more, only show the model
        # captures and checks so it doesn't just shuffle pieces around
        if self._material_balance(board) < 400:
            return moves  

        aggressive = []
        for move in moves:
            is_capture = board.is_capture(move)
            board.push(move)
            gives_check = board.is_check()
            board.pop()
            if is_capture or gives_check:
                aggressive.append(move)

        return aggressive if aggressive else moves

    def _position_key(self, board: chess.Board) -> str:
        # first 4 fields of FEN = position without move counters
        return " ".join(board.fen().split()[:4])

    def _get_non_repeating_moves(self, board: chess.Board, moves: list) -> list:
        non_repeating = []
        for move in moves:
            board.push(move)
            key = self._position_key(board)
            board.pop()
            if key not in self._seen_positions:
                non_repeating.append(move)
        return non_repeating

    def _record_move(self, board: chess.Board, move: chess.Move) -> None:
        """Record the position we are moving INTO for future repetition checks."""
        board.push(move)
        self._seen_positions.add(self._position_key(board))
        board.pop()

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        # reset position history at the start of each new game.
        if fen == chess.STARTING_FEN or board.fullmove_number == 1:
            self._seen_positions = set()

        # record current position so future moves can avoid returning here
        self._seen_positions.add(self._position_key(board))

        # take the checkmate if it's there
        mate_move = self._find_checkmate(board)
        if mate_move:
            self._record_move(board, mate_move)
            return mate_move.uci()

        # # always promote to queen
        promo_move = self._find_promotion(board)
        if promo_move:
            self._record_move(board, promo_move)
            return promo_move.uci()

        # filter out moves that repeat positions we've already been in
        non_repeating = self._get_non_repeating_moves(board, legal_moves)
        moves_to_offer = non_repeating if non_repeating else legal_moves

        # if we're clearly winning, only show the model aggressive moves
        moves_to_offer = self._filter_winning_moves(board, moves_to_offer)

        # build the prompt in the same format as training
        legal_moves_str = " ".join(m.uci() for m in moves_to_offer)
        input_text = f"chess: {fen} legal: {legal_moves_str}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8,
                num_beams=5,
                early_stopping=True,
            )

        predicted_move = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()
        print(f"Model predicted: '{predicted_move}'")  # add this

        # check the predicted move is actually legal before returning it
        try:
            move = chess.Move.from_uci(predicted_move)
            if move in board.legal_moves:
                self._record_move(board, move)
                return predicted_move
        except Exception:
            pass

        # fallback to a random non-repeating move if the model failed
        fallback_pool = non_repeating if non_repeating else legal_moves
        fallback = random.choice(fallback_pool)
        self._record_move(board, fallback)
        return fallback.uci()
