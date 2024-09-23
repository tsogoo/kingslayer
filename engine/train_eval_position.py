import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import sqlite3


# Define the neural network model
class PositionEvaluation(nn.Module):
    def __init__(self):
        super(PositionEvaluation, self).__init__()
        # Input layer size: piece-square table (12 x 64)
        self.input_size = 12 * 64  # 768 features for piece-square interaction

        # Define hidden layers (mimicking NNUE architecture)
        self.fc1 = nn.Linear(self.input_size, 512)  # First hidden layer
        self.fc2 = nn.Linear(512, 256)              # Second hidden layer
        self.fc3 = nn.Linear(256, 128)              # Third hidden layer
        self.output_layer = nn.Linear(128, 1)       # Output layer

        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        output = self.output_layer(x)
        return output


def extract_features(board):
    piece_types = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING]
    board_representation = np.zeros((12, 64), dtype=np.float32)

    for i, piece_type in enumerate(piece_types):
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # If the piece is of a given type and color, place a 1 at that square
                if piece.piece_type == piece_type:
                    if piece.color == chess.WHITE:
                        board_representation[i, square] = 1.0  # White piece
                    else:
                        board_representation[i + 6, square] = 1.0  # Black piece
    # Flatten to a 1D array of size 768
    return board_representation.flatten()


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, criterion, and optimizer
model = PositionEvaluation().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10


# Load training data from SQLite database
def load_training_data():
    conn = sqlite3.connect('training.db')
    cursor = conn.cursor()
    cursor.execute('SELECT fen, score FROM training_data')
    data = cursor.fetchall()
    conn.close()
    return data


dataset = load_training_data()

def train_model():
    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0

        for fen, target_eval in dataset:
            board = chess.Board(fen)
            # Extract features and move them to the appropriate device
            features = torch.tensor(extract_features(board), dtype=torch.float32).unsqueeze(0).to(device)
            target = torch.tensor([target_eval], dtype=torch.float32).to(device)

            # Forward pass
            prediction = model(features)
            loss = criterion(prediction, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss for printing
            running_loss += loss.item()

        # Print the average loss for this epoch
        avg_loss = running_loss / len(dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}')

        # Save the model at the end of each epoch
        torch.save(model.state_dict(), f'position_eval_epoch_{epoch+1}.pt')
        print('Model saved!')


class KingslayerEvaluator():
    def __init__(self):
        self.model = PositionEvaluation()
        self.model.load_state_dict(torch.load('position_eval_epoch_10.pt'))
        self.model.to(device)
        self.model.eval()

    def set_engine(self, engine):
        self.engine = self.engine

    def get_evaluation(self, board):
        features = torch.tensor(extract_features(board), dtype=torch.float32).unsqueeze(0).to(device)
        prediction = self.model(features)
        return prediction.item()


def load_model_and_evaluate_position(fen):
    model.load_state_dict(torch.load('position_eval_epoch_10.pt'))
    model.eval()

    board = chess.Board(fen)
    features = torch.tensor(extract_features(board), dtype=torch.float32).unsqueeze(0).to(device)
    prediction = model(features)
    return prediction.item()

# train_model()
# eval = load_model_and_evaluate_position('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
# print(eval)
