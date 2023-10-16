from qmath import Quaternion
import nn

class QuaternionDemo:
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

    def display_basic_operations(self):
        print(f"q1: {self.q1}")
        print(f"q2: {self.q2}")
        print(f"q1 * q2: {self.q1 * self.q2}")
        print(f"q1 + q2: {self.q1 + self.q2}")

    def display_angle_axis_representation(self, q):
        angle, axis = q.to_angle_axis()
        print(f"Angle-axis representation of {q}: Angle = {angle}, Axis = {axis}")

def demonstrate_quaternion_operations():
    q1 = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(2, 3, 4, 5)

    demo = QuaternionDemo(q1, q2)
    demo.display_basic_operations()
    demo.display_angle_axis_representation(q1)
    # Add more demonstrations as needed.

def train_rl_model():
    """Function to train RL model using Quaternion Neural Network."""
    nn.train_dqn()

if __name__ == "__main__":
    print("Demonstrating Quaternion Operations:")
    demonstrate_quaternion_operations()
    
    print("\nTraining RL Model using Quaternion Neural Network:")
    train_rl_model()
