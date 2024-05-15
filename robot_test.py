from robot_arm.robot import *
from engine.helper import *

class Test:
   
    def __init__(self):

        # for testing
        self.chess_engine_helper = ChessEngineHelper()
        fen = "4k3/7p/8/8/8/8/8/4K2R w KQkq"
        self.chess_engine_helper.initialize_board(fen)
        
        # robot handler
        self.robot_handler = RobotHandler()
    def move(self, best_move):
        moves = []
        
        f, t = self.chess_engine_helper.get_position(best_move)
        xs, ys = f[0], f[1]
        xd, yd = t[0], t[1]
        
        # source
        x_s, y_s = self.get_figure_actual_position(xs, ys, True)
        
        # destination
        is_occupied = self.chess_engine_helper.is_occupied(best_move)
        x_d, y_d = self.get_figure_actual_position(xd, yd, is_occupied)

        # king castle
        castling, king_castling = self.chess_engine_helper.is_castling(best_move)
        
        # king castling move
        if castling:
            # king
            moves.append(RobotMove(RobotTask.Take, x_s, y_s))
            moves.append(RobotMove(RobotTask.Place, x_d, y_d))
            
            # rook
            x_s, y_s = self.get_figure_actual_position(xs+3 if king_castling else xs-4, ys, True)
            x_d, y_d = self.get_figure_actual_position(xs+1 if king_castling else xs-1, ys, True)
            moves.append(RobotMove(RobotTask.Take, x_s, y_s))
            moves.append(RobotMove(RobotTask.Place, x_d, y_d))
        else:
            if is_occupied:
                moves.append(RobotMove(RobotTask.Take, x_d, y_d))
                moves.append(RobotMove(RobotTask.Out))
            
            is_promotion = self.chess_engine_helper.is_promotion(best_move)
            # promotion move, pawn become queen
            if is_promotion:
                # pawn
                moves.append(RobotMove(RobotTask.Take, x_s, y_s))
                moves.append(RobotMove(RobotTask.Out))
                
                # queen
                moves.append(RobotMove(RobotTask.In))
                moves.append(RobotMove(RobotTask.Place, x_d, y_d))
            else:
                # normal move
                moves.append(RobotMove(RobotTask.Take, x_s, y_s))
                moves.append(RobotMove(RobotTask.Place, x_d, y_d))

        self.robot_handler.move(moves)

    def get_figure_actual_position(self, x, y, is_occupied=False):
        if is_occupied:
            # TODO implement
            return x*46+23, y*46+23    
        return x*46+23, y*46+23
    

# testing
# t = Test()
# t.move("e1g1") # castle
# t.move("h7h8") # promotion
# t.move("h1h7")
    
# r = Robot()
# print(r.calc_position(0, 300, 60, 0, 150, 40.19))
# print(r.calc_position(0, 300, 80, 0, 300, 60))