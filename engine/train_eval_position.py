import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import sqlite3

class PositionEvaluation(nn.Module):
    def __init__(self, input_size=768, hidden1_size=512, hidden2_size=256, hidden3_size=128, output_size=1, switch_king_feature=False):
        super(PositionEvaluation, self).__init__()
        self.switch_king_feature = switch_king_feature

        # Adjust input size if switch king feature is enabled
        if self.switch_king_feature:
            input_size += 1  # Adding one more feature for switch king

        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.output_layer = nn.Linear(hidden3_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        output = self.output_layer(x)
        return output

def extract_features(board, switch_king_feature=False):
    piece_types = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING]
    board_representation = np.zeros((12, 64), dtype=np.float32)

    for i, piece_type in enumerate(piece_types):
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if piece.piece_type == piece_type:
                    if piece.color == chess.WHITE:
                        board_representation[i, square] = 1.0
                    else:
                        board_representation[i + 6, square] = 1.0

    features = board_representation.flatten()

    # Add switch king feature if enabled
    if switch_king_feature:
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        switch_king_value = 1.0 if white_king_square < black_king_square else 0.0
        features = np.append(features, switch_king_value)

    return features

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, criterion, and optimizer
model = PositionEvaluation(switch_king_feature=True).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)



# Load training data from SQLite database
def load_training_data():
    conn = sqlite3.connect("training.db")
    cursor = conn.cursor()
    cursor.execute("SELECT fen, score FROM training_data")
    data = cursor.fetchall()
    conn.close()
    return data


# Train the model
def train_model():
    # Training loop
    num_epochs = 500
    dataset = load_training_data()
    for epoch in range(num_epochs):
        for fen, target_eval in dataset:
            board = chess.Board(fen)
            features = torch.tensor(extract_features(board, switch_king_feature=True), dtype=torch.float32).unsqueeze(0).to(device)
            target = torch.tensor([target_eval], dtype=torch.float32).unsqueeze(0).to(device)  # Reshape target to match prediction

            # Forward pass
            prediction = model(features)
            loss = criterion(prediction, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the loss every epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        # Save the model
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'position_eval{epoch+1}.pt')

    # Save the model
    print('Model saved!')


# Load model and evaluate position
def load_model_and_evaluate_position(fen, model):
    print("loading model and evaluating position")

    board = chess.Board(fen)
    features = torch.tensor(extract_features(board, switch_king_feature=True), dtype=torch.float32).unsqueeze(0).to(device)
    prediction = model(features)
    return prediction.item()


class KingslayerEvaluator:

    def __init__(self):
        self.model = PositionEvaluation(switch_king_feature=True).to(device)
        self.model.load_state_dict(torch.load("position_eval21.pt"))
        self.model.eval()

    def setEngine(self, engine):
        self.engine = 1

    def get_evaluation(self, board):
        print("loading model and evaluating position")
        return load_model_and_evaluate_position(board.fen(), self.model)


# Main execution
if __name__ == "__main__":
    train_model()
    model.load_state_dict(torch.load("position_eval81.pt"))
    model.eval()
    print("Good for white")
    eval = load_model_and_evaluate_position("6k1/5ppp/8/8/8/6Q1/6P1/6K1 w - - 0 1", model)
    print(eval)
    print("Good for black")
    eval = load_model_and_evaluate_position("8/8/5k2/7K/5p2/7q/8/8 w - - 24 43", model)
    print(eval)
