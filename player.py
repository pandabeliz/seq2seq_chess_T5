import chess
import torch
import random
from typing import Optional
from transformers import T5TokenizerFast, T5ForConditionalGeneration

# Assumes Player base class is imported from your assignment environment
# from player import Player


class TransformerPlayer(Player):
    def __init__(self, name: str, model_path: str = "belpekkan/chess_T5_seq2seq"):
        #v4
        # model_path now has a default value -> satisfies the assignment requirement:
        # TransformerPlayer("Student") works without any extra arguments
        super().__init__(name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # --- Repetition penalty: persistent position history ---
        self._seen_positions: set = set()

    # ===================================================================
    # OVERRIDE 1 — Checkmate (2-ply)
    #
    # Phase 1 (mate-in-1): Check every legal move. If any delivers
    # immediate checkmate, return it instantly.
    #
    # Phase 2 (mate-in-2): For each candidate move, check whether
    # EVERY legal opponent reply allows us a checkmate-in-1 on the
    # next turn. If yes, that move forces mate in 2 — play it.
    #
    # Cost: O(our_moves × opponent_moves) board pushes.
    # In a lone-king endgame the opponent has ≤ 8 moves, so this is
    # effectively free compared to the T5 forward pass.
    # ===================================================================
    def _find_checkmate(self, board: chess.Board) -> Optional[chess.Move]:
        # ---- Phase 1: mate-in-1 ----------------------------------------
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()

        # ---- Phase 2: mate-in-2 ----------------------------------------
        for move in board.legal_moves:
            board.push(move)

            # Skip stalemate — never hand the opponent a stalemate escape
            if board.is_stalemate():
                board.pop()
                continue

            opponent_moves = list(board.legal_moves)

            # If the opponent has no moves and it's not checkmate, it's
            # stalemate — already filtered above, but guard here too.
            if not opponent_moves:
                board.pop()
                continue

            all_replies_mated = True
            for opp_move in opponent_moves:
                board.push(opp_move)

                # Check whether we have a mate-in-1 from this position
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

    # ===================================================================
    # OVERRIDE 2 — Promotion
    # If any legal move is a pawn promotion, always promote to queen.
    # If multiple promotion squares exist, prefer the one that gives check
    # (most aggressive), otherwise just return the first one found.
    # ===================================================================
    def _find_promotion(self, board: chess.Board) -> Optional[chess.Move]:
        promotions = [
            m for m in board.legal_moves
            if m.promotion == chess.QUEEN
        ]
        if not promotions:
            return None
        # Prefer a promotion that also gives check
        for move in promotions:
            board.push(move)
            gives_check = board.is_check()
            board.pop()
            if gives_check:
                return move
        return promotions[0]

    # ===================================================================
    # OVERRIDE 3 — Winning position: prefer captures and checks
    # When our material advantage exceeds WINNING_ADVANTAGE_CP, restrict
    # the moves offered to the model to only:
    #   (a) moves that give check, OR
    #   (b) moves that capture material
    # This prevents aimless repositioning when we are clearly winning.
    # Falls back to all moves if no such moves exist (very rare endgames).
    # ===================================================================

    # ------------------------------------------------------------------
    # Material values used for Override 3 (winning position detection).
    # Standard centipawn values — only piece type matters, not colour.
    # ------------------------------------------------------------------
    PIECE_VALUES = {
        chess.PAWN:   100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK:   500,
        chess.QUEEN:  900,
        chess.KING:     0,   # never counted in material sums
    }

    # Centipawn advantage threshold above which Override 3 activates.
    # 400cp ≈ roughly a rook up — clearly winning territory.
    WINNING_ADVANTAGE_CP = 400


    def _material_balance(self, board: chess.Board) -> int:
        """
        Returns material balance in centipawns from the perspective of the
        side to move (positive = we are winning, negative = we are losing).
        """
        score = 0
        for piece_type, value in self.PIECE_VALUES.items():
            score += value * len(board.pieces(piece_type, board.turn))
            score -= value * len(board.pieces(piece_type, not board.turn))
        return score

    def _filter_winning_moves(
        self, board: chess.Board, moves: list
    ) -> list:
        if self._material_balance(board) < self.WINNING_ADVANTAGE_CP:
            return moves  # not clearly winning — don't filter

        aggressive = []
        for move in moves:
            is_capture = board.is_capture(move)
            board.push(move)
            gives_check = board.is_check()
            board.pop()
            if is_capture or gives_check:
                aggressive.append(move)

        return aggressive if aggressive else moves

    # ===================================================================
    # Repetition penalty helpers
    # ===================================================================
    def _position_key(self, board: chess.Board) -> str:
        """
        Fingerprint of a board position that ignores move counters.
        Uses the first 4 space-separated fields of the FEN:
          piece placement / active colour / castling rights / en-passant
        """
        return " ".join(board.fen().split()[:4])

    def _get_non_repeating_moves(
        self, board: chess.Board, moves: list
    ) -> list:
        """
        Filter out moves that would return the game to a position we have
        already visited, preventing cycling behaviour.
        """
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

    # ===================================================================
    # get_move — main entry point called by the tournament
    # ===================================================================
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        # ------------------------------------------------------------------
        # Reset position history at the start of each new game.
        # Detected by the standard starting FEN or fullmove number == 1.
        # ------------------------------------------------------------------
        if fen == chess.STARTING_FEN or board.fullmove_number == 1:
            self._seen_positions = set()

        # Record current position so future moves can avoid returning here
        self._seen_positions.add(self._position_key(board))

        # ==================================================================
        # OVERRIDE 1: Deliver checkmate immediately if available (2-ply)
        # ==================================================================
        mate_move = self._find_checkmate(board)
        if mate_move:
            self._record_move(board, mate_move)
            return mate_move.uci()

        # ==================================================================
        # OVERRIDE 2: Promote to queen immediately if available
        # ==================================================================
        promo_move = self._find_promotion(board)
        if promo_move:
            self._record_move(board, promo_move)
            return promo_move.uci()

        # ------------------------------------------------------------------
        # Apply repetition filter to the full legal move list
        # ------------------------------------------------------------------
        non_repeating = self._get_non_repeating_moves(board, legal_moves)
        moves_to_offer = non_repeating if non_repeating else legal_moves

        # ==================================================================
        # OVERRIDE 3: In clearly winning positions, only show the model
        # aggressive moves (checks + captures) to prevent aimless orbiting
        # ==================================================================
        moves_to_offer = self._filter_winning_moves(board, moves_to_offer)

        # ------------------------------------------------------------------
        # Build prompt and query the model.
        # Mirror the training format exactly: "chess: <FEN> legal: e2e4 ..."
        # ------------------------------------------------------------------
        legal_moves_str = " ".join(m.uci() for m in moves_to_offer)
        input_text = f"chess: {fen} legal: {legal_moves_str}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,       # must match training
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

        # ------------------------------------------------------------------
        # Validate — the predicted move must be legal on the FULL board
        # (not just the filtered list) for safety.
        # ------------------------------------------------------------------
        try:
            move = chess.Move.from_uci(predicted_move)
            if move in board.legal_moves:
                self._record_move(board, move)
                return predicted_move
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Fallback: random move from the non-repeating pool, then full pool.
        # Avoids returning None and incurring an illegal move penalty.
        # ------------------------------------------------------------------
        fallback_pool = non_repeating if non_repeating else legal_moves
        fallback = random.choice(fallback_pool)
        self._record_move(board, fallback)
        return fallback.uci()
