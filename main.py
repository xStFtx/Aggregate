import argparse
import logging
from qmath import Quaternion
import nn  # Import your neural network module here

class QuaternionOperationDemonstrator:
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

    def display_basic_operations(self):
        """Display basic quaternion operations."""
        print(f"q1: {self.q1}")
        print(f"q2: {self.q2}")
        print(f"q1 * q2: {self.q1 * self.q2}")
        print(f"q1 + q2: {self.q1 + self.q2}")

    def display_angle_axis_representation(self, q):
        """Display angle-axis representation of a quaternion."""
        angle, axis = q.to_angle_axis()
        print(f"Angle-axis representation of {q}: Angle = {angle}, Axis = {axis}")

def demonstrate_quaternion_operations():
    """Demonstrate quaternion operations."""
    q1 = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(2, 3, 4, 5)

    demo = QuaternionOperationDemonstrator(q1, q2)
    demo.display_basic_operations()
    demo.display_angle_axis_representation(q1)
    # Add more demonstrations as needed.

def train_rl_model(logging_level):
    """Train an RL model using Quaternion Neural Network."""
    # Configure logging
    logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        nn.train_dqn()
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quaternion Operations and RL Training")
    parser.add_argument("--log-level", choices=["INFO", "DEBUG"], default="INFO",
                        help="Logging level (INFO or DEBUG)")
    args = parser.parse_args()

    # Set the logging level
    log_level = getattr(logging, args.log_level)
    
    print("Demonstrating Quaternion Operations:")
    demonstrate_quaternion_operations()
    
    print("\nTraining RL Model using Quaternion Neural Network:")
    train_rl_model(log_level)
